## Overview

This project implements a full pipeline for molecular and crystal density prediction using graph neural networks. The workflow covers:

- Crystal structure collection (COD)
- Molecular property extraction (Reaxys)
- CIF → SMILES parsing
- SMILES → 3D conformer generation
- Density computation
- Database construction
- Graph feature engineering (Atom Graph + Line Graph)
- ALIGNN model training and evaluation
- Transfer learning for nitrogen-containing compounds

The goal is to build a physically interpretable, geometry-aware deep learning model for density prediction.

---

## Motivation

Density is a fundamental physical property closely related to:

- Molecular mass
- Molecular packing efficiency
- Intermolecular interactions
- Hydrogen bonding ability
- Structural rigidity

Traditional computational approaches (e.g., DFT) are accurate but expensive. Graph neural networks provide a scalable alternative for learning structure–property relationships directly from atomistic representations.

The ALIGNN architecture is chosen because it explicitly models:

- Bond distances
- Bond angles
- Three-body geometric interactions

This makes it particularly suitable for geometry-sensitive properties like density.

---

## Data Sources

### 1. Crystallography Open Database (COD)

- Open-access crystallographic data
- CIF files parsed to extract:
  - Lattice parameters
  - Cell volume
  - Z values
  - Molecular formula
- Density calculated using:

  ρ = (Z × M) / (N_A × V)

### 2. Reaxys (Nitrogen-containing molecules)

- SMILES strings
- Experimental density values
- Molecular descriptors

All data are structured into SQLite databases for traceability and reproducibility.

---

## Project Structure

```

Molecular-Density-ALIGNN/
│
├── COD_Density_Pretraining/
│   ├── ALIGNN_Model.py
│   ├── MOL_to_Graph.py
│   ├── Database-to-TrainData.py
│   ├── Train.py
│
├── NDensity_Transfer_Model/
│   ├── ALIGNN_Model.py
│   ├── Train_N.py
│   ├── MOL_to_Graph.py
│   ├── Database-to-TrainData.py
│
├── Data/
│   ├── COD_csv/
│   ├── CODid_SMILES.csv
│   ├── reaxys_N.csv
│
├── Data_Integration/
│   ├── COD-SMILES-to-MOL.py
│   ├── CODcsv-to-Database.py
│   ├── CrystalDensity_Calculation.py
│   ├── Database_Analysis.py
│   ├── Nitrogen-containing-Molecules-to-Database.py
│   ├── Reaxys-SMILES-to-MOL.py
│   ├── SMILES-to-Database.py
│
├── SMILES2MOL/
│   └── Custom robust SMILES → 3D MOL library
│
└── README.md

```

---

## Graph Construction Methodology

### 1. Atom Graph

- Nodes: Atoms
- Edges: Chemical bonds (bidirectional)
- Node features (physically interpretable):
  - Atomic number (normalized)
  - Electronegativity
  - Covalent radius
  - Atomic mass
  - Formal charge
  - Aromaticity
  - Hybridization
  - Hydrogen count

- Edge features:
  - Bond type
  - Ring membership
  - Conjugation
  - Bond length
  - Bond length²
  - 1 / bond length

---

### 2. Line Graph (Angle Graph)

- Nodes: Bonds from atom graph
- Edges: Bond pairs sharing a common atom
- Features:
  - Bond angle (radians)
  - Cosine(angle)
  - Adjacent bond lengths
  - Mean bond length

This explicitly models three-body geometric interactions.

---

### 3. Global Features

- Molecular weight
- Atom count
- Dipole moment (Gasteiger charge-based)
- Hydrogen bond donor count
- Hydrogen bond acceptor count
- Per-atom energy

---

## ALIGNN Architecture

The Atomistic Line Graph Neural Network (ALIGNN) consists of:

1. Atom Graph Message Passing
2. Line Graph Message Passing
3. Edge-Gated Convolution
4. Residual Connections
5. Global Pooling
6. Fully Connected Readout Layer

Key characteristics:

- Explicit angle modeling
- Edge gating mechanism
- Simultaneous node and edge updates
- Geometry-aware learning

---

## Training Pipeline

The training framework includes:

- SmoothL1Loss (Huber loss)
- AdamW optimizer
- Gradient clipping
- Learning rate scheduler (ReduceLROnPlateau)
- Early stopping
- Full experiment logging
- Automatic model checkpoint saving

Dataset split:

- 80% training
- 10% validation
- 10% testing

---

## Results

Model performance is evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

The ALIGNN model demonstrates strong predictive performance, especially compared to bond-only GNN baselines, highlighting the importance of explicit angle modeling for density prediction.

(Exact numerical results are printed during training and saved in experiment logs.)

---

## How to Run

### 1. Install Requirements

```

pip install torch
pip install torch-geometric
pip install rdkit
pip install numpy pandas scikit-learn matplotlib

```

Note: RDKit installation via conda is recommended.

---

### 2. Pretraining on COD

```

cd COD_Density_Pretraining
python Train.py

```

---

### 3. Transfer Learning (Nitrogen Dataset)

```

cd NDensity_Transfer_Model
python Train_N.py

```

---

## Database Design

Two main databases:

### COD_Database.db

Contains:

- CellData (73 fields)
- SMILES
- Density
- Molecules (3D structures + optimization logs)
- Precheck diagnostics

### reaxys_N.db

Contains:

- compoundsN
- Molecules
- Precheck

All structures undergo robust SMILES validation and 3D embedding before training.

---

## Key Features of This Project

- Full end-to-end pipeline
- Physically interpretable feature engineering
- Robust SMILES-to-3D conversion system
- Explicit three-body geometric modeling
- Transfer learning setup
- Reproducible training configuration
- Automated statistical normalization

---

## Requirements

- torch
- torch-geometric
- torch-scatter
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm
- rdkit

---

## License

MIT License

---

## Educational Purpose

This project was developed as a university-level coursework project in:

- Machine Learning for Materials Science
- Computational Chemistry
- Graph Neural Networks
- Structure–Property Modeling

---

## Author

Zhenhao Huang

---



