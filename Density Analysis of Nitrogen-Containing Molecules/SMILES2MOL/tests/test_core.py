"""
test_core.py - Standalone test script without pytest

Features:
1. Displays test results via direct console output
2. Shows detailed input, output, and diagnostic information
3. Uses optional color highlighting for success/failure
4. Automatically calculates pass rate
"""

import sys
from rdkit import Chem

# Add src directory to path
sys.path.insert(0, "src")

from smiles2mol import (
    build_mol_from_smiles,
    MolBuildConfig,
    DEFAULT_ORGANIC_CONFIG,
    METAL_COMPLEX_CONFIG,
    CHARGED_SPECIES_CONFIG,
    DEBUG_LENIENT_CONFIG,
)


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


try:
    import colorama
    colorama.init()
except ImportError:
    Colors.GREEN = ""
    Colors.RED = ""
    Colors.YELLOW = ""
    Colors.BLUE = ""
    Colors.BOLD = ""
    Colors.UNDERLINE = ""
    Colors.END = ""


class TestRunner:
    """Simple test runner."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0

    def run_test(self, name, test_func):
        print(f"\n{Colors.BOLD}Test: {name}{Colors.END}")
        print("-" * 60)

        self.total += 1
        try:
            result = test_func()
            if result:
                print(f"{Colors.GREEN}PASS{Colors.END}")
                self.passed += 1
            else:
                print(f"{Colors.RED}FAIL{Colors.END}")
                self.failed += 1
        except Exception as e:
            print(f"{Colors.RED}EXCEPTION: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
            self.failed += 1

    def print_summary(self):
        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
        print(f"{Colors.BOLD}Test Summary{Colors.END}")
        print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")

        if self.failed == 0:
            print(f"{Colors.GREEN}All tests passed! ({self.passed}/{self.total}){Colors.END}")
        else:
            print(f"{Colors.RED}Some tests failed.{Colors.END}")
            print(f"Passed: {self.passed}")
            print(f"Failed: {self.failed}")
            print(f"Total: {self.total}")
            print(f"Pass rate: {self.passed / self.total * 100:.1f}%")

        print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")


def test_basic_organic_molecule():
    """Test construction of a basic organic molecule."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    print(f"Input SMILES: {smiles}")
    print("Using config: DEFAULT_ORGANIC_CONFIG")

    mol, report = build_mol_from_smiles(smiles, DEFAULT_ORGANIC_CONFIG)

    print(f"Decision: {report['decision']}")
    print(f"Reasons: {report['reasons']}")

    if mol is None:
        print("Error: Molecule is None")
        return False

    if report["decision"] != "accepted":
        print(f"Error: Expected 'accepted', got '{report['decision']}'")
        return False

    if not isinstance(mol, Chem.Mol):
        print("Error: Returned object is not Chem.Mol")
        return False

    if mol.GetNumAtoms() == 0:
        print("Error: Molecule has zero atoms")
        return False

    print(f"Molecule constructed successfully. Atoms: {mol.GetNumAtoms()}")
    print(f"Conformers: {mol.GetNumConformers()}")
    print(f"Optimization results: {report['opt_results']}")
    return True


def test_metal_complex_with_organic_config():
    """Metal complex with organic config (should be rejected)."""
    smiles = "[Fe+2]C(C)C"
    print(f"Input SMILES: {smiles}")
    print("Using config: DEFAULT_ORGANIC_CONFIG")

    mol, report = build_mol_from_smiles(smiles, DEFAULT_ORGANIC_CONFIG)

    print(f"Decision: {report['decision']}")
    print(f"Reasons: {report['reasons']}")

    if mol is not None:
        print("Error: Expected None (rejected), but molecule was built")
        return False

    if report["decision"] != "rejected":
        print(f"Error: Expected 'rejected', got '{report['decision']}'")
        return False

    return True


def test_invalid_smiles():
    """Test invalid SMILES handling."""
    smiles = "invalid_smiles_string"
    print(f"Input SMILES: {smiles}")

    mol, report = build_mol_from_smiles(smiles, DEFAULT_ORGANIC_CONFIG)

    print(f"Decision: {report['decision']}")
    print(f"Reasons: {report['reasons']}")

    if mol is not None:
        print("Error: Expected None for invalid SMILES")
        return False

    if report["decision"] != "rejected":
        print(f"Error: Expected 'rejected', got '{report['decision']}'")
        return False

    return True


def main():
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}SMILES2MOL Test Script{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")

    import rdkit
    print(f"RDKit version: {rdkit.__version__}")

    runner = TestRunner()

    tests = [
        ("Basic Organic Molecule", test_basic_organic_molecule),
        ("Metal Complex Rejection (Organic Config)", test_metal_complex_with_organic_config),
        ("Invalid SMILES", test_invalid_smiles),
    ]

    for name, test_func in tests:
        runner.run_test(name, test_func)

    runner.print_summary()

    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())