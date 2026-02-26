# part 1/2
"""
database_to_graphs_alignn.py

Read molecule records from a COD-like SQLite database, convert them to ALIGNN-style
PyG Data using build_alignn_data_from_mol, normalize global features, and save as a .pt dataset file.
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
        Query the database for molecules that have density information.

        Returns a list of dicts with keys:
        - id: molecule ID
        - mol_text: molecule text (mol block or SMILES)
        - energy: energy value
        - density: density value
        """
        query = """
        SELECT 
            m.ID,
            m.mol_text,
            m.opt_results,
            d.density_clean
        FROM Molecules m
        LEFT JOIN compoundsN d ON m.ID = d.ID
        WHERE m.mol_text IS NOT NULL 
          AND m.mol_text != 'None'
          AND m.opt_results IS NOT NULL
          AND m.opt_results != '[]'
          AND d.density_clean IS NOT NULL
        """
        if limit:
            query += f" LIMIT {limit}"

        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        molecules: List[Dict] = []

        for row in rows:
            try:
                mol_id, mol_text, opt_results_str, density = row

                # Parse optimization results (expected as JSON list)
                opt_results = json.loads(opt_results_str)
                if not opt_results:
                    continue

                # Try to get energy value (default to first entry's 'energy')
                energy = opt_results[0].get('energy')
                if energy is None:
                    # If first entry has no energy, search entire list
                    for entry in opt_results:
                        if 'energy' in entry:
                            energy = entry['energy']
                            break

                if energy is None:
                    continue  # cannot find energy value -> skip

                molecules.append({
                    'id': int(mol_id),
                    'mol_text': mol_text,
                    'energy': float(energy),
                    'density': float(density)
                })

            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                print(f"[WARN] Failed to parse row id={row[0]}: {exc}")
                continue

        print(f"[DB] Read {len(molecules)} molecules with density and energy from the database")
        return molecules

    def mol_text_to_mol(self, mol_text: str) -> Optional[Chem.Mol]:
        """
        Convert molecule text to an RDKit Mol object.

        Parameters:
            mol_text: molecule text (mol block or SMILES)

        Returns:
            RDKit Mol object, or None if conversion fails
        """
        try:
            # Try parsing as mol block first
            mol = Chem.MolFromMolBlock(mol_text, removeHs=False)

            if mol is None:
                # If that fails, try parsing as SMILES
                mol = Chem.MolFromSmiles(mol_text)
                if mol is not None:
                    # Add hydrogens for SMILES
                    mol = Chem.AddHs(mol)

            return mol
        except Exception as e:
            print(f"[WARN] mol_text -> Mol conversion failed: {e}")
            return None

    def build_graph_dataset(
            self,
            molecules: List[Dict],
            max_examples: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Convert a list of molecule dictionaries into a list of ALIGNN-format PyG Data objects.

        Parameters:
            molecules: list of molecule dicts
            max_examples: maximum number to process

        Returns:
            list of PyG Data objects
        """
        graph_data_list: List[torch.Tensor] = []
        total = len(molecules) if max_examples is None else min(len(molecules), max_examples)

        print(f"[BUILD] Converting up to {total} molecules into ALIGNN graph structures...")

        for idx, entry in enumerate(molecules[:total]):
            mid = entry['id']
            mol_text = entry['mol_text']
            energy = entry['energy']
            density = entry['density']

            # Convert to RDKit Mol
            mol = self.mol_text_to_mol(mol_text)
            if mol is None:
                print(f"  [SKIP] Could not parse mol_text id={mid}")
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Use the ALIGNN data builder
                    data = build_alignn_data_from_mol(
                        mol_id=mid,
                        mol=mol,
                        energy=float(energy),
                        density=float(density),
                        dipole=None,  # computed automatically if needed
                        hbd=None,     # computed automatically if needed
                        hba=None,     # computed automatically if needed
                        add_hydrogens=True
                    )

                if data is None:
                    print(f"  [SKIP] build_alignn_data_from_mol returned None id={mid}")
                    continue

                # Check required ALIGNN fields exist
                if not hasattr(data, 'line_graph_edge_index') or not hasattr(data, 'line_graph_edge_attr'):
                    print(f"  [SKIP] Missing required ALIGNN features id={mid}")
                    continue

                # Validate global features if present
                if hasattr(data, 'u'):
                    try:
                        u_valid = torch.isfinite(data.u).all()
                        if not u_valid:
                            print(f"  [SKIP] Global features contain NaN/Inf id={mid}")
                            continue
                    except Exception:
                        print(f"  [SKIP] Global feature check failed id={mid}")
                        continue

                # Append to list
                graph_data_list.append(data)

                if len(graph_data_list) % 100 == 0:
                    print(f"  Built {len(graph_data_list)} graphs")

            except Exception as e:
                print(f"  [ERR] Failed to build graph for molecule id={mid}: {e}")
                continue

        print(f"[BUILD] Done. Successfully built {len(graph_data_list)} ALIGNN graphs.")
        return graph_data_list

    def compute_dataset_statistics(self, graph_data_list: List[torch.Tensor]) -> Dict:
        """
        Compute dataset statistics.

        Parameters:
            graph_data_list: list of graph Data objects

        Returns:
            dictionary containing statistics
        """
        if not graph_data_list:
            return {}

        stats = {
            'num_graphs': len(graph_data_list),
            'atom_features': {},
            'edge_features': {},
            'line_edge_features': {},
            'global_features': {},
        }

        # Collect atom feature statistics
        atom_feats = torch.cat([g.x for g in graph_data_list], dim=0)
        stats['atom_features']['mean'] = atom_feats.mean(dim=0).tolist()
        stats['atom_features']['std'] = atom_feats.std(dim=0).tolist()
        stats['atom_features']['min'] = atom_feats.min(dim=0)[0].tolist()
        stats['atom_features']['max'] = atom_feats.max(dim=0)[0].tolist()

        # Collect edge feature statistics
        edge_feats = torch.cat([g.edge_attr for g in graph_data_list], dim=0)
        stats['edge_features']['mean'] = edge_feats.mean(dim=0).tolist()
        stats['edge_features']['std'] = edge_feats.std(dim=0).tolist()

        # Collect line-graph edge feature statistics
        line_edge_feats = torch.cat(
            [g.line_graph_edge_attr for g in graph_data_list if hasattr(g, 'line_graph_edge_attr')], dim=0)
        if len(line_edge_feats) > 0:
            stats['line_edge_features']['mean'] = line_edge_feats.mean(dim=0).tolist()
            stats['line_edge_features']['std'] = line_edge_feats.std(dim=0).tolist()

        # Collect global feature statistics
        global_feats = torch.stack([g.u for g in graph_data_list], dim=0)
        stats['global_features']['mean'] = global_feats.mean(dim=0).tolist()
        stats['global_features']['std'] = global_feats.std(dim=0).tolist()

        # Collect density statistics
        densities = torch.stack([g.y for g in graph_data_list], dim=0)
        stats['density'] = {
            'mean': float(densities.mean()),
            'std': float(densities.std()),
            'min': float(densities.min()),
            'max': float(densities.max()),
        }

        # Atom count statistics
        num_atoms = [g.num_atoms.item() if hasattr(g, 'num_atoms') else g.x.size(0) for g in graph_data_list]
        stats['num_atoms'] = {
            'mean': float(np.mean(num_atoms)),
            'std': float(np.std(num_atoms)),
            'min': int(min(num_atoms)),
            'max': int(max(num_atoms)),
        }

        return stats
# part 2/2
    def normalize_features(self, graph_data_list: List[torch.Tensor],
                           stats: Optional[Dict] = None) -> Tuple[List[torch.Tensor], Dict]:
        """
        Normalize features.

        Parameters:
            graph_data_list: list of graph Data objects
            stats: precomputed statistics (if None, computed automatically)

        Returns:
            (normalized_graphs, normalization_parameters)
        """
        if not graph_data_list:
            return graph_data_list, {}

        # Compute stats if not provided
        if stats is None:
            stats = self.compute_dataset_statistics(graph_data_list)

        # Extract normalization parameters
        atom_mean = torch.tensor(stats['atom_features']['mean'], dtype=torch.float32)
        atom_std = torch.tensor(stats['atom_features']['std'], dtype=torch.float32)
        atom_std = torch.where(atom_std < 1e-8, torch.ones_like(atom_std), atom_std)

        edge_mean = torch.tensor(stats['edge_features']['mean'], dtype=torch.float32)
        edge_std = torch.tensor(stats['edge_features']['std'], dtype=torch.float32)
        edge_std = torch.where(edge_std < 1e-8, torch.ones_like(edge_std), edge_std)

        global_mean = torch.tensor(stats['global_features']['mean'], dtype=torch.float32)
        global_std = torch.tensor(stats['global_features']['std'], dtype=torch.float32)
        global_std = torch.where(global_std < 1e-8, torch.ones_like(global_std), global_std)

        # Apply normalization
        for data in graph_data_list:
            # Normalize atom features
            data.x = (data.x - atom_mean) / atom_std

            # Normalize edge features
            data.edge_attr = (data.edge_attr - edge_mean) / edge_std

            # Normalize global features
            data.u = (data.u - global_mean) / global_std

            # Normalize line-graph edge features if present
            if hasattr(data, 'line_graph_edge_attr') and 'line_edge_features' in stats:
                line_edge_mean = torch.tensor(stats['line_edge_features']['mean'], dtype=torch.float32)
                line_edge_std = torch.tensor(stats['line_edge_features']['std'], dtype=torch.float32)
                line_edge_std = torch.where(line_edge_std < 1e-8, torch.ones_like(line_edge_std), line_edge_std)
                data.line_graph_edge_attr = (data.line_graph_edge_attr - line_edge_mean) / line_edge_std

        # Create normalization parameters dict
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
        """
        Save dataset to disk.

        Parameters:
            graph_data_list: list of graph Data objects
            output_path: output file path
            stats: dataset statistics
            norm_params: normalization parameters
        """
        if not graph_data_list:
            print("[SAVE] No graph data to save")
            return

        # Create output directory if needed
        dirname = os.path.dirname(output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        # Save graph data
        torch.save(graph_data_list, output_path)
        print(f"[SAVE] Saved {len(graph_data_list)} graphs to {output_path}")

        # Save statistics if provided
        if stats is not None:
            stats_path = os.path.splitext(output_path)[0] + "_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"[SAVE] Saved stats to {stats_path}")

        # Save normalization parameters if provided
        if norm_params is not None:
            norm_path = os.path.splitext(output_path)[0] + "_norm_params.json"
            with open(norm_path, 'w', encoding='utf-8') as f:
                json.dump(norm_params, f, indent=2, ensure_ascii=False)
            print(f"[SAVE] Saved normalization parameters to {norm_path}")

        # Quick load verification
        try:
            loaded = torch.load(output_path)
            print(f"[SAVE] Load verification successful: loaded {len(loaded)} graphs")
        except Exception as e:
            print(f"[SAVE] Load verification failed: {e}")

    def process_and_save(self, limit: int = None, output_path: str = "alignn_graph_dataset.pt"):
        """
        Main processing pipeline: query DB, build graphs, normalize and save.

        Parameters:
            limit: maximum number of molecules to process
            output_path: output .pt file path
        """
        try:
            # Connect to DB
            self.connect()
            print(f"[PROC] Querying database (limit={limit}) ...")

            # Get molecules
            molecules = self.get_molecules_with_density(limit=limit)
            if not molecules:
                print("[PROC] No molecule data found")
                return

            # Build graphs
            graphs = self.build_graph_dataset(molecules, max_examples=limit)
            if not graphs:
                print("[PROC] Failed to build any graphs")
                return

            # Compute statistics
            stats = self.compute_dataset_statistics(graphs)
            print("\n[STATS] Dataset statistics:")
            print(f"  num_graphs: {stats['num_graphs']}")
            print(f"  num_atoms range: {stats['num_atoms']['min']} - {stats['num_atoms']['max']}")
            print(f"  density range: {stats['density']['min']:.4f} - {stats['density']['max']:.4f}")
            print(f"  atom feature dim: {len(stats['atom_features']['mean'])}")
            print(f"  edge feature dim: {len(stats['edge_features']['mean'])}")
            print(f"  global feature dim: {len(stats['global_features']['mean'])}")

            # Normalize features
            graphs, norm_params = self.normalize_features(graphs, stats)

            # Check first few graphs
            print("\n[CHECK] Inspecting first 2 graphs:")
            for i, g in enumerate(graphs[:2]):
                print(f"  Graph {i + 1}:")
                print(f"    num_atoms: {g.num_atoms.item() if hasattr(g, 'num_atoms') else g.x.size(0)}")
                print(f"    num_edges: {g.edge_index.shape[1]}")
                print(f"    num_line_graph_edges: {g.line_graph_edge_index.shape[1] if hasattr(g, 'line_graph_edge_index') else 0}")
                print(f"    global_features: {g.u.tolist()}")
                print(f"    target_density: {g.y.item():.6f}")

            # Save dataset
            self.save_dataset(graphs, output_path, stats=stats, norm_params=norm_params)

        except Exception as e:
            print(f"[PROC] Error occurred during processing: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.close()


def print_alignn_data_summary(data_list: List[torch.Tensor], num_samples: int = 3):
    """
    Print a summary of an ALIGNN dataset.

    Parameters:
        data_list: list of Data objects
        num_samples: number of samples to print
    """
    if not data_list:
        print("Data list is empty")
        return

    print("\n" + "=" * 80)
    print("ALIGNN Dataset Summary")
    print("=" * 80)
    print(f"Total samples: {len(data_list)}")

    # Aggregate counts
    total_atoms = 0
    total_edges = 0
    total_line_edges = 0

    for i, data in enumerate(data_list[:num_samples]):
        print(f"\nSample {i + 1} ({data.mol_id.item() if hasattr(data, 'mol_id') else 'N/A'}):")
        print(f"  num_atoms: {data.x.shape[0]}")
        print(f"  atom_feature_dim: {data.x.shape[1]}")
        print(f"  atom_graph_num_edges: {data.edge_index.shape[1]}")
        print(f"  edge_feature_dim: {data.edge_attr.shape[1]}")
        print(f"  line_graph_num_edges: {data.line_graph_edge_index.shape[1]}")
        print(f"  line_graph_edge_feature_dim: {data.line_graph_edge_attr.shape[1]}")
        print(f"  global_features: {data.u.tolist()}")
        print(f"  target_density: {data.y.item():.6f}")

        total_atoms += data.x.shape[0]
        total_edges += data.edge_index.shape[1]
        total_line_edges += data.line_graph_edge_index.shape[1]

    # Print overall stats for inspected samples
    print(f"\nOverall (for first {min(num_samples, len(data_list))} samples):")
    print(f"  avg_num_atoms: {total_atoms / min(num_samples, len(data_list)):.2f}")
    print(f"  avg_num_edges: {total_edges / min(num_samples, len(data_list)):.2f}")
    print(f"  avg_num_line_graph_edges: {total_line_edges / min(num_samples, len(data_list)):.2f}")

    # Confirm feature shapes using first sample
    if len(data_list) > 0:
        sample_data = data_list[0]
        print(f"\nFeature shape confirmation:")
        print(f"  x shape: {sample_data.x.shape}")
        print(f"  edge_index shape: {sample_data.edge_index.shape}")
        print(f"  edge_attr shape: {sample_data.edge_attr.shape}")
        print(f"  line_graph_edge_index shape: {sample_data.line_graph_edge_index.shape}")
        print(f"  line_graph_edge_attr shape: {sample_data.line_graph_edge_attr.shape}")
        print(f"  u shape: {sample_data.u.shape}")
        print(f"  y shape: {sample_data.y.shape}")

    print("=" * 80 + "\n")


def main():
    """Main entry point"""
    DB_PATH = "../Data/reaxys_N.db"
    OUTPUT_PATH = "TrainData/alignn_graph_dataset_N.pt"
    DEBUG_LIMIT = 10  # Set to None to process all, or a number to debug

    print("=" * 60)
    print("Database -> ALIGNN graph dataset conversion")
    print("=" * 60)

    # Create processor
    processor = DatabaseToGraphsALIGNN(DB_PATH, device=torch.device("cpu"))

    # Process and save
    processor.process_and_save(limit=DEBUG_LIMIT, output_path=OUTPUT_PATH)

    # Load and print summary
    try:
        loaded_data = torch.load(OUTPUT_PATH)
        print_alignn_data_summary(loaded_data, num_samples=3)
    except Exception as e:
        print(f"Failed to load dataset: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()