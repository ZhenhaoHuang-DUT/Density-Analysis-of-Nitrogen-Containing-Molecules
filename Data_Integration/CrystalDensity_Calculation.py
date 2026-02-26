import sqlite3
import re

# -----------------------------
# Constants
# -----------------------------
AVOGADRO = 0.602214076  # mol^-1 (scaled for Å^3 → cm^3 conversion)

# Atomic weights table (extendable)
ATOMIC_WEIGHTS = {
    "H": 1.008,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,

    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,

    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955908,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.921595,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,

    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90584,
    "Zr": 91.224,
    "Nb": 92.90637,
    "Mo": 95.95,
    "Tc": 98.0,
    "Ru": 101.07,
    "Rh": 102.90550,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.60,
    "I": 126.90447,
    "Xe": 131.293,

    "Cs": 132.90545196,
    "Ba": 137.327,
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    "Pm": 145.0,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.92535,
    "Dy": 162.500,
    "Ho": 164.93033,
    "Er": 167.259,
    "Tm": 168.93422,
    "Yb": 173.045,
    "Lu": 174.9668,

    "Hf": 178.49,
    "Ta": 180.94788,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.966569,
    "Hg": 200.592,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.98040,
    "Po": 209.0,
    "At": 210.0,
    "Rn": 222.0,

    "Fr": 223.0,
    "Ra": 226.0,
    "Ac": 227.0,
    "Th": 232.0377,
    "Pa": 231.03588,
    "U": 238.02891,
    "Np": 237.0,
    "Pu": 244.0,
    "Am": 243.0,
    "Cm": 247.0,
    "Bk": 247.0,
    "Cf": 251.0,
    "Es": 252.0,
    "Fm": 257.0,
    "Md": 258.0,
    "No": 259.0,
    "Lr": 266.0,

    "Rf": 267.0,
    "Db": 268.0,
    "Sg": 269.0,
    "Bh": 270.0,
    "Hs": 269.0,
    "Mt": 278.0,
    "Ds": 281.0,
    "Rg": 282.0,
    "Cn": 285.0,
    "Nh": 286.0,
    "Fl": 289.0,
    "Mc": 290.0,
    "Lv": 293.0,
    "Ts": 294.0,
    "Og": 294.0,
}


# -----------------------------
# Chemical formula parser
# Supported format example: Ca4 Mg4 O24 Si8
# -----------------------------
def parse_formula(formula: str):
    """
    Parse a chemical formula string.

    Returns:
        dict: {element: count}
        None: if parsing fails or unknown element is found
    """
    if formula is None:
        return None

    formula = formula.replace("-", " ").strip()

    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pattern, formula)

    if not matches:
        return None

    comp = {}
    for elem, num in matches:
        if elem not in ATOMIC_WEIGHTS:
            return None
        count = int(num) if num else 1
        comp[elem] = comp.get(elem, 0) + count

    return comp


def molar_mass(comp: dict):
    """
    Compute molar mass in g/mol.
    """
    return sum(ATOMIC_WEIGHTS[e] * n for e, n in comp.items())


# -----------------------------
# Main procedure
# -----------------------------
def main():
    conn = sqlite3.connect("../Data/COD_Database.db")
    cur = conn.cursor()

    # Create Density table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Density (
            ID INTEGER PRIMARY KEY,
            Density REAL
        )
    """)

    # Read CellData table
    cur.execute("""
        SELECT ID, vol, cellformula, formula, Z
        FROM CellData
    """)

    rows = cur.fetchall()
    inserted = 0
    skipped = 0

    for ID, vol, cellformula, formula, Z in rows:
        if vol is None:
            skipped += 1
            continue

        comp = None

        # Prefer cellformula
        if cellformula:
            comp = parse_formula(cellformula)
            if comp is None:
                skipped += 1
                continue
            M_cell = molar_mass(comp)

        # Fallback: formula × Z
        elif formula and Z:
            comp = parse_formula(formula)
            if comp is None:
                skipped += 1
                continue
            M_cell = molar_mass(comp) * Z

        else:
            skipped += 1
            continue

        # Density in g/cm^3
        density = (M_cell / AVOGADRO) / vol

        cur.execute("""
            INSERT OR REPLACE INTO Density (ID, Density)
            VALUES (?, ?)
        """, (ID, density))

        inserted += 1

    conn.commit()
    conn.close()

    print(f"Completed: inserted {inserted} records, skipped {skipped} records")


if __name__ == "__main__":
    main()