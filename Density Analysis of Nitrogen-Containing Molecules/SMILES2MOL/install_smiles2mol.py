# install_smiles2mol.py

import subprocess
import sys
import os


def install_smiles2mol() -> bool:
    """
    One-click installer for the local smiles2mol package.

    Usage:
        python install_smiles2mol.py
    """

    # Path to the smiles2mol project (modify if necessary)
    smiles2mol_path = "../SMILES2MOL"

    print(f"Installing smiles2mol from: {smiles2mol_path}")

    # Check if path exists
    if not os.path.exists(smiles2mol_path):
        print(f"ERROR: Path does not exist: {smiles2mol_path}")
        return False

    # Check for pyproject.toml
    if not os.path.exists(os.path.join(smiles2mol_path, "pyproject.toml")):
        print(f"ERROR: pyproject.toml not found in {smiles2mol_path}")
        return False

    try:
        # Install smiles2mol in editable mode
        cmd = [sys.executable, "-m", "pip", "install", "-e", smiles2mol_path]
        print(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Installation completed successfully.")
        print("Output:")
        print(result.stdout)

        # Verify installation
        test_install()

        return True

    except subprocess.CalledProcessError as e:
        print("Installation failed.")
        print("Error message:")
        print(e.stderr)
        return False

    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return False


def test_install() -> bool:
    """
    Test whether smiles2mol was installed successfully.
    """

    print("\n" + "=" * 50)
    print("Testing smiles2mol import...")

    try:
        from smiles2mol import build_mol_from_smiles, DEFAULT_ORGANIC_CONFIG

        print("Import successful.")

        # Simple functionality test
        smiles = "CCCC"
        mol, report = build_mol_from_smiles(smiles, DEFAULT_ORGANIC_CONFIG)

        print("Functionality test passed.")
        print(f"  Decision: {report['decision']}")

        if mol:
            print(f"  Number of atoms: {mol.GetNumAtoms()}")

        return True

    except ImportError as e:
        print(f"Import failed: {e}")
        return False

    except Exception as e:
        print(f"Functionality test failed: {e}")
        return False


def show_usage() -> None:
    """
    Display usage instructions after successful installation.
    """

    print("\n" + "=" * 50)
    print("Installation completed.")
    print("You can now use smiles2mol in other scripts as follows:\n")

    print(
        """from smiles2mol import build_mol_from_smiles, DEFAULT_ORGANIC_CONFIG

smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
mol, report = build_mol_from_smiles(smiles, DEFAULT_ORGANIC_CONFIG)

if mol:
    print("Molecule constructed successfully.")
"""
    )


if __name__ == "__main__":
    print("smiles2mol Installation Script")
    print("=" * 50)

    if install_smiles2mol():
        show_usage()
    else:
        print("\nInstallation failed. Please review the error messages above.")
        sys.exit(1)