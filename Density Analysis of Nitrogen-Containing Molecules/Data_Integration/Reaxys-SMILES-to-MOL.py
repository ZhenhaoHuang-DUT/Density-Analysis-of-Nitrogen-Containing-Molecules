from smiles2mol import build_mol_from_smiles, DEFAULT_ORGANIC_CONFIG
import os
import json
import sqlite3
from rdkit import Chem

# ==========================================================
# Configuration
# ==========================================================

DB_PATH = "../Data/reaxys_N.db"
BATCH_SIZE = 1000
MOL_DIR = "mol_files"

os.makedirs(MOL_DIR, exist_ok=True)


# ==========================================================
# Utility Functions
# ==========================================================

def j(x):
    """
    Convert any object to a JSON-safe string.
    Falls back to string conversion if serialization fails.
    """
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return json.dumps(str(x), ensure_ascii=False)


def mol_to_molblock_safe(mol):
    """
    Safely convert an RDKit Mol object to MolBlock format.
    Returns None if mol is None.
    """
    if mol is None:
        return None
    return Chem.MolToMolBlock(mol)


# ==========================================================
# Database Initialization
# ==========================================================

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Create Molecules table if it does not already exist
cur.execute("""
CREATE TABLE IF NOT EXISTS Molecules (
    ID INTEGER PRIMARY KEY,
    mol_text TEXT,
    decision TEXT,
    reasons TEXT,
    opt_results TEXT,
    exception TEXT
)
""")

# Create Precheck table if it does not already exist
cur.execute("""
CREATE TABLE IF NOT EXISTS Precheck (
    ID INTEGER PRIMARY KEY,
    smiles TEXT,
    success INTEGER,
    total_formal_charge INTEGER,
    contains_metal INTEGER,
    contains_organic_component INTEGER,
    has_radical INTEGER,
    has_isotopes INTEGER,
    has_unusual_valence INTEGER,
    action_recommendation TEXT
)
""")


# ==========================================================
# Batch Processing
# ==========================================================

offset = 0

while True:
    # Fetch BATCH_SIZE records per iteration
    cur.execute(
        f"SELECT ID, SMILES FROM compoundsN LIMIT {BATCH_SIZE} OFFSET {offset}"
    )
    rows = cur.fetchall()

    if not rows:
        break  # All records processed

    for idx, smiles in rows:
        try:
            mol, report = build_mol_from_smiles(
                smiles,
                DEFAULT_ORGANIC_CONFIG
            )
        except Exception as e:
            # Store error information if molecule construction fails
            cur.execute(
                "INSERT OR REPLACE INTO Molecules VALUES (?, ?, ?, ?, ?, ?)",
                (idx, None, "error", None, None, j(str(e)))
            )
            continue

        # ------------------------------------------------------
        # Store molecule information
        # ------------------------------------------------------

        mol_text = mol_to_molblock_safe(mol)

        decision = report.get("decision")
        reasons = j(report.get("reasons"))
        opt_results = j(report.get("opt_results"))
        exception = j(report.get("exception"))

        cur.execute(
            "INSERT OR REPLACE INTO Molecules VALUES (?, ?, ?, ?, ?, ?)",
            (idx, mol_text, decision, reasons, opt_results, exception)
        )

        # ------------------------------------------------------
        # Store precheck information (selected fields only)
        # ------------------------------------------------------

        pre = report.get("precheck")

        if pre is not None:
            cur.execute("""
            INSERT OR REPLACE INTO Precheck VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                idx,
                pre.smiles,
                pre.success,
                pre.total_formal_charge,
                pre.contains_metal,
                pre.contains_organic_component,
                pre.has_radical,
                pre.has_isotopes,
                pre.has_unusual_valence,
                pre.action_recommendation
            ))

    # Commit after processing each batch
    conn.commit()

    print(f"Processed {offset + len(rows)} records")

    offset += BATCH_SIZE


conn.close()

print("Completed: all molecules and reports have been saved.")