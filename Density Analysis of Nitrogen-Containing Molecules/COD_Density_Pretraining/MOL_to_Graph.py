"""
Implementation for constructing atomic graphs and bond line graphs
for the ALIGNN model.

Returned Data contains:
 - x: atomic features (N, node_feat_dim)
 - edge_index: atomic graph edge indices (2, E_atom)
 - edge_attr: atomic graph edge features (E_atom, edge_feat_dim)
 - line_graph_edge_index: bond line graph edge indices (2, E_line)
 - line_graph_edge_attr: bond line graph edge features (E_line, angle_feat_dim)
 - pos: atomic coordinates (N, 3)
 - u: global features
 - y: density target
"""

import os
import math
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


# Electronegativity (Pauling scale)
ELECTRONEGATIVITY = {
    'H': 2.20, 'He': 4.16, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
    'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 4.79, 'Na': 0.93, 'Mg': 1.31,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 3.24,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
    'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.00,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16,
    'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 2.60,
}

# Covalent radii (Å)
COVALENT_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
    'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
    'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
    'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
}


def compute_bond_angle_features(pos, edge_index, neighbors):
    """Compute bond angle features required by ALIGNN"""
    n_edges = edge_index.shape[1]
    angle_indices = []
    angle_features = []

    # Build edge index mapping
    edge_dict = {}
    for e_idx in range(n_edges):
        i, j = edge_index[:, e_idx].tolist()
        edge_dict[(i, j)] = e_idx
        edge_dict[(j, i)] = e_idx

    # Create angle features for each bond pair sharing a central atom
    for e1_idx in range(n_edges):
        i, j = edge_index[:, e1_idx].tolist()

        for k in neighbors[j]:
            if k == i:
                continue

            if (j, k) in edge_dict:
                e2_idx = edge_dict[(j, k)]

                v1 = pos[i] - pos[j]
                v2 = pos[k] - pos[j]

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 > 1e-6 and norm2 > 1e-6:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)

                    angle_indices.append([e1_idx, e2_idx])
                    angle_features.append([
                        angle,
                        cos_angle,
                        norm1,
                        norm2,
                        (norm1 + norm2) / 2.0,
                    ])

    if angle_indices:
        return np.array(angle_indices, dtype=np.int64).T, np.array(angle_features, dtype=np.float32)
    else:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 5, dtype=torch.float32)


def build_alignn_data_from_mol(
        mol_id: int = 0,
        mol: Chem.Mol = None,
        energy: float = None,
        density: float = None,
        dipole: float = None,
        hbd: int = None,
        hba: int = None,
        add_hydrogens: bool = True,
) -> Data:
    """
    Construct ALIGNN-compatible graph data from an RDKit molecule.

    Args:
        mol: RDKit molecule (must contain 3D conformer)
        energy: molecular energy
        density: target density
        add_hydrogens: whether to add hydrogens
    """

    if mol is None or energy is None or density is None:
        return None

    # Add hydrogens if required
    if add_hydrogens and not mol.HasProp('_has_hydrogens'):
        mol = Chem.AddHs(mol)

    n_atoms = mol.GetNumAtoms()
    if energy / n_atoms > 100:
        return None

    # Extract 3D coordinates
    try:
        conf = mol.GetConformer()
        pos = np.zeros((n_atoms, 3), dtype=np.float32)
        for i in range(n_atoms):
            p = conf.GetAtomPosition(i)
            pos[i] = [p.x, p.y, p.z]
    except:
        return None

    # ---------------- Atomic Features (9 selected features) ----------------
    node_feats = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()

        z_norm = atomic_num / 100.0
        en = ELECTRONEGATIVITY.get(symbol, 2.0)
        en_norm = en / 4.0
        rad = COVALENT_RADII.get(symbol, 1.0)
        rad_norm = rad / 2.0
        mass = atom.GetMass() / 200.0

        valence = atom.GetTotalValence()
        total_electrons = atomic_num
        valence_ratio = valence / total_electrons if total_electrons > 0 else 0.0

        formal_charge = atom.GetFormalCharge() / 2.0
        is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0

        hybrid = atom.GetHybridization()
        hybrid_map = {
            Chem.HybridizationType.SP: 0.0,
            Chem.HybridizationType.SP2: 0.5,
            Chem.HybridizationType.SP3: 1.0,
        }
        hybrid_val = hybrid_map.get(hybrid, 0.25)

        num_h = atom.GetTotalNumHs() / 4.0

        node_feats.append([
            z_norm,
            en_norm,
            rad_norm,
            mass,
            valence_ratio,
            formal_charge,
            is_aromatic,
            hybrid_val,
            num_h,
        ])

    x = torch.tensor(np.array(node_feats, dtype=np.float32))

    # ---------------- Atomic Graph Edges (chemical bonds) ----------------
    row, col = [], []
    edge_features = []
    neighbors = [[] for _ in range(n_atoms)]

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        row.extend([i, j])
        col.extend([j, i])

        bt = bond.GetBondType()
        bt_val = 1.0 if bt == Chem.BondType.SINGLE else \
            2.0 if bt == Chem.BondType.DOUBLE else \
                3.0 if bt == Chem.BondType.TRIPLE else \
                    1.5 if bt == Chem.BondType.AROMATIC else 1.0

        in_ring = 1.0 if bond.IsInRing() else 0.0
        is_conjugated = 1.0 if bond.GetIsConjugated() else 0.0
        dist = np.linalg.norm(pos[i] - pos[j])

        edge_feat = [
            bt_val,
            in_ring,
            is_conjugated,
            dist,
            dist ** 2,
            1.0 / dist if dist > 1e-6 else 0.0,
        ]

        edge_features.extend([edge_feat, edge_feat])

        neighbors[i].append(j)
        neighbors[j].append(i)

    if len(row) == 0:
        for i in range(n_atoms):
            row.append(i)
            col.append(i)
            edge_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features, dtype=np.float32))

    # Do NOT call to_undirected() since edges were manually duplicated

    # ---------------- Line Graph (bond-bond graph) ----------------
    line_graph_edge_index, line_graph_edge_attr = compute_bond_angle_features(
        pos, edge_index, neighbors
    )

    # ---------------- Global Features ----------------
    if dipole is None:
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = np.zeros(n_atoms)
            for i, atom in enumerate(mol.GetAtoms()):
                charge = atom.GetDoubleProp('_GasteigerCharge')
                if not np.isfinite(charge):
                    charge = 0.0
                charges[i] = charge

            dipole_vec = np.zeros(3)
            for i in range(n_atoms):
                dipole_vec += charges[i] * pos[i]
            dipole = float(np.linalg.norm(dipole_vec))
        except:
            return None

    if hbd is None or hba is None:
        try:
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
        except:
            hbd, hba = 0, 0

    mol_weight = rdMolDescriptors.CalcExactMolWt(mol)

    global_feats = torch.tensor([[
        float(energy) / n_atoms,
        float(dipole),
        float(hbd),
        float(hba),
        float(mol_weight),
        float(n_atoms),
    ]], dtype=torch.float32)

    y = torch.tensor([float(density)], dtype=torch.float32)

    # ---------------- Construct PyG Data object ----------------
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        line_graph_edge_index=torch.tensor(line_graph_edge_index, dtype=torch.long),
        line_graph_edge_attr=torch.tensor(line_graph_edge_attr, dtype=torch.float32),
        pos=torch.tensor(pos, dtype=torch.float32),
        u=global_feats,
        y=y,
        num_atoms=torch.tensor([n_atoms], dtype=torch.long),
    )

    if mol_id is not None:
        data.mol_id = torch.tensor([mol_id], dtype=torch.long)

    return data