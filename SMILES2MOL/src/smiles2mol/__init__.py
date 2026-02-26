# src/smiles2mol/__init__.py

"""
smiles2mol

A research-oriented toolkit for:
- SMILES validity diagnosis
- Metal / organic composition analysis
- Robust SMILES → RDKit Mol construction
"""

from .core import build_mol_from_smiles
from .config import (
    MolBuildConfig,
    DEFAULT_ORGANIC_CONFIG,
    METAL_COMPLEX_CONFIG,
    CHARGED_SPECIES_CONFIG,
    DEBUG_LENIENT_CONFIG,
)

__all__ = [
    "build_mol_from_smiles",
    "MolBuildConfig",
    "DEFAULT_ORGANIC_CONFIG",
    "METAL_COMPLEX_CONFIG",
    "CHARGED_SPECIES_CONFIG",
    "DEBUG_LENIENT_CONFIG",
]

__version__ = "0.1.0"
