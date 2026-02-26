"""
Read molecular records from a COD-like SQLite database, convert them into
ALIGNN-style PyG Data objects using build_data_from_mol_alignn,
normalize global features, and save as a .pt dataset file.
"""

import sqlite3
import json
import os
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
import warnings
from MOL_to_Graph import build_alignn_data_from_mol


class DatabaseToGraphsALIGNN:
    def __init__(self, db_path: str, device: Optional[torch.device] = None):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor = None
        self.device = device or torch.device("cpu")

    def connect(self):
        """Connect to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        print(f"[DB] Connected to {self.db_path}")

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            print("[DB] Connection closed")

    def get_molecules_with_density(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Query molecules that contain valid density and optimization results.

        Returns a list of dictionaries with keys:
        - id: molecule ID
        - mol_text: molecular representation (mol block or SMILES)
        - energy: energy value
        - density: density value
        """
        query = """
        SELECT 
            m.ID,
            m.mol_text,
            m.opt_results,
            d.Density
        FROM Molecules m
        LEFT JOIN Density d ON m.ID = d.ID
        WHERE m.mol_text IS NOT NULL 
          AND m.mol_text != 'None'
          AND m.opt_results IS NOT NULL
          AND m.opt_results != '[]'
          AND d.Density IS NOT NULL
        """
        if limit:
            query += f" LIMIT {limit}"

        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        molecules: List[Dict] = []

        for row in rows:
            try:
                mol_id, mol_text, opt_results_str, density = row
                opt_results = json.loads(opt_results_str)
                if not opt_results:
                    continue

                energy = opt_results[0].get('energy')
                if energy is None:
                    for entry in opt_results:
                        if 'energy' in entry:
                            energy = entry['energy']
                            break

                if energy is None:
                    continue

                molecules.append({
                    'id': int(mol_id),
                    'mol_text': mol_text,
                    'energy': float(energy),
                    'density': float(density)
                })

            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                print(f"[WARN] Failed to parse row id={row[0]}: {exc}")
                continue

        print(f"[DB] Retrieved {len(molecules)} molecules with density and energy")
        return molecules

    def mol_text_to_mol(self, mol_text: str) -> Optional[Chem.Mol]:
        """
        Convert molecular text into an RDKit Mol object.

        Args:
            mol_text: mol block or SMILES string

        Returns:
            RDKit Mol object or None if parsing fails.
        """
        try:
            mol = Chem.MolFromMolBlock(mol_text, removeHs=False)

            if mol is None:
                mol = Chem.MolFromSmiles(mol_text)
                if mol is not None:
                    mol = Chem.AddHs(mol)

            return mol
        except Exception as e:
            print(f"[WARN] Mol parsing failed: {e}")
            return None

    def build_graph_dataset(
            self,
            molecules: List[Dict],
            max_examples: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Convert molecule dictionaries into ALIGNN PyG Data objects.
        """
        graph_data_list: List[torch.Tensor] = []
        total = len(molecules) if max_examples is None else min(len(molecules), max_examples)

        print(f"[BUILD] Converting up to {total} molecules into ALIGNN graphs...")

        for idx, entry in enumerate(molecules[:total]):
            mid = entry['id']
            mol_text = entry['mol_text']
            energy = entry['energy']
            density = entry['density']

            mol = self.mol_text_to_mol(mol_text)
            if mol is None:
                print(f"  [SKIP] Failed to parse molecule id={mid}")
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    data = build_alignn_data_from_mol(
                        mol_id=mid,
                        mol=mol,
                        energy=float(energy),
                        density=float(density),
                        dipole=None,
                        hbd=None,
                        hba=None,
                        add_hydrogens=True
                    )

                if data is None:
                    print(f"  [SKIP] build_alignn_data_from_mol returned None id={mid}")
                    continue

                if not hasattr(data, 'line_graph_edge_index') or not hasattr(data, 'line_graph_edge_attr'):
                    print(f"  [SKIP] Missing required ALIGNN features id={mid}")
                    continue

                if hasattr(data, 'u'):
                    try:
                        if not torch.isfinite(data.u).all():
                            print(f"  [SKIP] Global features contain NaN/Inf id={mid}")
                            continue
                    except Exception:
                        print(f"  [SKIP] Global feature validation failed id={mid}")
                        continue

                graph_data_list.append(data)

                if len(graph_data_list) % 100 == 0:
                    print(f"  Built {len(graph_data_list)} graphs")

            except Exception as e:
                print(f"  [ERR] Graph construction failed id={mid}: {e}")
                continue

        print(f"[BUILD] Completed. Successfully built {len(graph_data_list)} ALIGNN graphs.")
        return graph_data_list

    def compute_dataset_statistics(self, graph_data_list: List[torch.Tensor]) -> Dict:
        """Compute dataset statistics."""
        if not graph_data_list:
            return {}

        stats = {
            'num_graphs': len(graph_data_list),
            'atom_features': {},
            'edge_features': {},
            'line_edge_features': {},
            'global_features': {},
        }

        atom_feats = torch.cat([g.x for g in graph_data_list], dim=0)
        stats['atom_features']['mean'] = atom_feats.mean(dim=0).tolist()
        stats['atom_features']['std'] = atom_feats.std(dim=0).tolist()
        stats['atom_features']['min'] = atom_feats.min(dim=0)[0].tolist()
        stats['atom_features']['max'] = atom_feats.max(dim=0)[0].tolist()

        edge_feats = torch.cat([g.edge_attr for g in graph_data_list], dim=0)
        stats['edge_features']['mean'] = edge_feats.mean(dim=0).tolist()
        stats['edge_features']['std'] = edge_feats.std(dim=0).tolist()

        line_edge_feats = torch.cat(
            [g.line_graph_edge_attr for g in graph_data_list if hasattr(g, 'line_graph_edge_attr')], dim=0)
        if len(line_edge_feats) > 0:
            stats['line_edge_features']['mean'] = line_edge_feats.mean(dim=0).tolist()
            stats['line_edge_features']['std'] = line_edge_feats.std(dim=0).tolist()

        global_feats = torch.stack([g.u for g in graph_data_list], dim=0)
        stats['global_features']['mean'] = global_feats.mean(dim=0).tolist()
        stats['global_features']['std'] = global_feats.std(dim=0).tolist()

        densities = torch.stack([g.y for g in graph_data_list], dim=0)
        stats['density'] = {
            'mean': float(densities.mean()),
            'std': float(densities.std()),
            'min': float(densities.min()),
            'max': float(densities.max()),
        }

        num_atoms = [g.num_atoms.item() if hasattr(g, 'num_atoms') else g.x.size(0) for g in graph_data_list]
        stats['num_atoms'] = {
            'mean': float(np.mean(num_atoms)),
            'std': float(np.std(num_atoms)),
            'min': int(min(num_atoms)),
            'max': int(max(num_atoms)),
        }

        return stats

    def normalize_features(self, graph_data_list: List[torch.Tensor],
                           stats: Optional[Dict] = None) -> Tuple[List[torch.Tensor], Dict]:
        """Normalize features using dataset statistics."""
        if not graph_data_list:
            return graph_data_list, {}

        if stats is None:
            stats = self.compute_dataset_statistics(graph_data_list)

        atom_mean = torch.tensor(stats['atom_features']['mean'], dtype=torch.float32)
        atom_std = torch.tensor(stats['atom_features']['std'], dtype=torch.float32)
        atom_std = torch.where(atom_std < 1e-8, torch.ones_like(atom_std), atom_std)

        edge_mean = torch.tensor(stats['edge_features']['mean'], dtype=torch.float32)
        edge_std = torch.tensor(stats['edge_features']['std'], dtype=torch.float32)
        edge_std = torch.where(edge_std < 1e-8, torch.ones_like(edge_std), edge_std)

        global_mean = torch.tensor(stats['global_features']['mean'], dtype=torch.float32)
        global_std = torch.tensor(stats['global_features']['std'], dtype=torch.float32)
        global_std = torch.where(global_std < 1e-8, torch.ones_like(global_std), global_std)

        for data in graph_data_list:
            data.x = (data.x - atom_mean) / atom_std
            data.edge_attr = (data.edge_attr - edge_mean) / edge_std
            data.u = (data.u - global_mean) / global_std

            if hasattr(data, 'line_graph_edge_attr') and 'line_edge_features' in stats:
                line_edge_mean = torch.tensor(stats['line_edge_features']['mean'], dtype=torch.float32)
                line_edge_std = torch.tensor(stats['line_edge_features']['std'], dtype=torch.float32)
                line_edge_std = torch.where(line_edge_std < 1e-8, torch.ones_like(line_edge_std), line_edge_std)
                data.line_graph_edge_attr = (data.line_graph_edge_attr - line_edge_mean) / line_edge_std

        norm_params = {
            'atom_mean': atom_mean.tolist(),
            'atom_std': atom_std.tolist(),
            'edge_mean': edge_mean.tolist(),
            'edge_std': edge_std.tolist(),
            'global_mean': global_mean.tolist(),
            'global_std': global_std.tolist(),
        }

        if 'line_edge_features' in stats:
            norm_params['line_edge_mean'] = stats['line_edge_features']['mean']
            norm_params['line_edge_std'] = stats['line_edge_features']['std']

        print("[NORM] Feature normalization completed")
        return graph_data_list, norm_params

    def save_dataset(self, graph_data_list: List[torch.Tensor], output_path: str,
                     stats: Optional[Dict] = None, norm_params: Optional[Dict] = None):
        """Save dataset and related metadata."""
        if not graph_data_list:
            print("[SAVE] No graph data to save")
            return

        dirname = os.path.dirname(output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        torch.save(graph_data_list, output_path)
        print(f"[SAVE] Saved {len(graph_data_list)} graphs to {output_path}")

        if stats is not None:
            stats_path = os.path.splitext(output_path)[0] + "_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"[SAVE] Saved statistics to {stats_path}")

        if norm_params is not None:
            norm_path = os.path.splitext(output_path)[0] + "_norm_params.json"
            with open(norm_path, 'w', encoding='utf-8') as f:
                json.dump(norm_params, f, indent=2, ensure_ascii=False)
            print(f"[SAVE] Saved normalization parameters to {norm_path}")

        try:
            loaded = torch.load(output_path)
            print(f"[SAVE] Reload validation successful: {len(loaded)} graphs loaded")
        except Exception as e:
            print(f"[SAVE] Reload validation failed: {e}")

    def process_and_save(self, limit: int = None, output_path: str = "alignn_graph_dataset.pt"):
        """Main pipeline: query DB → build graphs → normalize → save."""
        try:
            self.connect()
            print(f"[PROC] Querying database (limit={limit}) ...")

            molecules = self.get_molecules_with_density(limit=limit)
            if not molecules:
                print("[PROC] No molecule data found")
                return

            graphs = self.build_graph_dataset(molecules, max_examples=limit)
            if not graphs:
                print("[PROC] No graphs were constructed")
                return

            stats = self.compute_dataset_statistics(graphs)
            print("\n[STATS] Dataset statistics:")
            print(f"  Number of graphs: {stats['num_graphs']}")
            print(f"  Atom count range: {stats['num_atoms']['min']} - {stats['num_atoms']['max']}")
            print(f"  Density range: {stats['density']['min']:.4f} - {stats['density']['max']:.4f}")
            print(f"  Atom feature dimension: {len(stats['atom_features']['mean'])}")
            print(f"  Edge feature dimension: {len(stats['edge_features']['mean'])}")
            print(f"  Global feature dimension: {len(stats['global_features']['mean'])}")

            graphs, norm_params = self.normalize_features(graphs, stats)

            print("\n[CHECK] Inspecting first 2 graphs:")
            for i, g in enumerate(graphs[:2]):
                print(f"  Graph {i + 1}:")
                print(f"    Num atoms: {g.num_atoms.item() if hasattr(g, 'num_atoms') else g.x.size(0)}")
                print(f"    Num edges: {g.edge_index.shape[1]}")
                print(f"    Num line edges: {g.line_graph_edge_index.shape[1]}")
                print(f"    Global features: {g.u.tolist()}")
                print(f"    Target density: {g.y.item():.6f}")

            self.save_dataset(graphs, output_path, stats=stats, norm_params=norm_params)

        except Exception as e:
            print(f"[PROC] Error during processing: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.close()


def print_alignn_data_summary(data_list: List[torch.Tensor], num_samples: int = 3):
    """Print a summary of the ALIGNN dataset."""
    if not data_list:
        print("Dataset is empty")
        return

    print("\n" + "=" * 80)
    print("ALIGNN Dataset Summary")
    print("=" * 80)
    print(f"Total samples: {len(data_list)}")

    total_atoms = 0
    total_edges = 0
    total_line_edges = 0

    for i, data in enumerate(data_list[:num_samples]):
        print(f"\nSample {i + 1} ({data.mol_id.item() if hasattr(data, 'mol_id') else 'N/A'}):")
        print(f"  Num atoms: {data.x.shape[0]}")
        print(f"  Atom feature dim: {data.x.shape[1]}")
        print(f"  Num edges: {data.edge_index.shape[1]}")
        print(f"  Edge feature dim: {data.edge_attr.shape[1]}")
        print(f"  Num line edges: {data.line_graph_edge_index.shape[1]}")
        print(f"  Line edge feature dim: {data.line_graph_edge_attr.shape[1]}")
        print(f"  Global features: {data.u.tolist()}")
        print(f"  Target density: {data.y.item():.6f}")

        total_atoms += data.x.shape[0]
        total_edges += data.edge_index.shape[1]
        total_line_edges += data.line_graph_edge_index.shape[1]

    print(f"\nOverall statistics (first {num_samples} samples):")
    print(f"  Avg atoms: {total_atoms / num_samples:.2f}")
    print(f"  Avg edges: {total_edges / num_samples:.2f}")
    print(f"  Avg line edges: {total_line_edges / num_samples:.2f}")

    sample_data = data_list[0]
    print(f"\nFeature shape check:")
    print(f"  x shape: {sample_data.x.shape}")
    print(f"  edge_index shape: {sample_data.edge_index.shape}")
    print(f"  edge_attr shape: {sample_data.edge_attr.shape}")
    print(f"  line_graph_edge_index shape: {sample_data.line_graph_edge_index.shape}")
    print(f"  line_graph_edge_attr shape: {sample_data.line_graph_edge_attr.shape}")
    print(f"  u shape: {sample_data.u.shape}")
    print(f"  y shape: {sample_data.y.shape}")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    DB_PATH = "../Data/COD_Database.db"
    OUTPUT_PATH = "TrainData/alignn_graph_dataset.pt"
    DEBUG_LIMIT = None

    print("=" * 60)
    print("Database -> ALIGNN Graph Conversion")
    print("=" * 60)

    processor = DatabaseToGraphsALIGNN(DB_PATH, device=torch.device("cpu"))
    processor.process_and_save(limit=DEBUG_LIMIT, output_path=OUTPUT_PATH)

    try:
        loaded_data = torch.load(OUTPUT_PATH)
        print_alignn_data_summary(loaded_data, num_samples=3)
    except Exception as e:
        print(f"Failed to load dataset: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()