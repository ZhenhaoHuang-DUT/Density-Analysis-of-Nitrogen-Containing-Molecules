"""
csv_to_sqlite_db.py

Read a CSV containing chemical data and write to a SQLite .db file.

Features:
- Auto-detects the density column (case-insensitive) or accept user-specified column name.
- If density is a range like "0.142 - 0.379", the script computes the midpoint and saves that numeric value.
- If density is a single numeric value (e.g. "6.2"), it saves it as-is.
- Adds an `id` column (INTEGER) starting from 1 (or user-specified start) and saves table to SQLite.
- Handles missing/non-numeric density values by storing NULL.

Usage:
    python csv_to_sqlite_db.py input.csv output.db --table compounds

Optional arguments:
    --density-col    Provide exact name of the density column if auto-detection fails.
    --start-id       The integer to start the id column from (default 1).
    --table          The table name to write into the SQLite DB (default "compounds").
"""

import re
import argparse
import sqlite3
import math
import sys
from typing import Optional

import pandas as pd

# =========================
# CONFIG (edit here)
# =========================

CONFIG = {
    # input / output
    "csv_path": "../Data/reaxys_N.csv",
    "db_path": "../Data/reaxys_N.db",

    # database
    "table": "compoundsN",
    "start_id": 1,

    # column handling
    "density_col": "Density: Density [g·cm-3]",
}


def find_density_column(columns):
    """Return the first column name that likely refers to density (case-insensitive)."""
    for col in columns:
        if 'density' in col.lower():
            return col
    # fallback: look for unit-like substrings
    for col in columns:
        if 'g' in col.lower() and 'cm' in col.lower():
            return col
    return None


float_rx = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def parse_density(value) -> Optional[float]:
    """Parse density string and return float.

    Handles forms like:
      - "0.142 - 0.379"  -> returns midpoint (0.2605)
      - "0.142-0.379"
      - "0.2605"
      - "0.128 - 0.435"
      - empty / NaN -> returns None
      - any non-numeric content: attempt to extract numbers; if none, return None
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == '':
        return None

    # unify different dash characters
    s = s.replace('\u2013', '-')  # en dash
    s = s.replace('\u2014', '-')  # em dash
    s = s.replace('\u2212', '-')  # minus sign

    # If there's a dash indicating a range, split on dash
    if '-' in s:
        parts = [p.strip() for p in s.split('-') if p.strip() != '']
        nums = []
        for p in parts:
            m = float_rx.search(p)
            if m:
                try:
                    nums.append(float(m.group()))
                except ValueError:
                    pass
        if len(nums) >= 2:
            low, high = nums[0], nums[1]
            if math.isfinite(low) and math.isfinite(high):
                return (low + high) / 2.0

    # Generic extraction of numeric tokens
    all_nums = float_rx.findall(s)
    if not all_nums:
        return None
    try:
        floats = [float(x) for x in all_nums]
    except Exception:
        return None
    if len(floats) == 1:
        return floats[0]
    return sum(floats) / len(floats)


def df_sql_type(dtype) -> str:
    """Map pandas dtype to SQLite type."""
    if pd.api.types.is_integer_dtype(dtype):
        return 'INTEGER'
    if pd.api.types.is_float_dtype(dtype):
        return 'REAL'
    if pd.api.types.is_bool_dtype(dtype):
        return 'INTEGER'
    return 'TEXT'


def create_table_with_schema(conn: sqlite3.Connection, table: str, df: pd.DataFrame):
    """Create table with id INTEGER PRIMARY KEY and columns matching df (excluding id)."""
    cols_sql = []
    for col in df.columns:
        if col == 'id':
            continue
        col_safe = col.replace('"', '""')
        dtype = df_sql_type(df[col].dtype)
        cols_sql.append(f'"{col_safe}" {dtype}')

    create_sql = (
        f'CREATE TABLE IF NOT EXISTS "{table}" '
        f'("id" INTEGER PRIMARY KEY, {", ".join(cols_sql)});'
    )
    conn.execute(create_sql)
    conn.commit()


def insert_dataframe(conn: sqlite3.Connection, table: str, df: pd.DataFrame):
    """Insert all rows from df into the named table. Assumes the table already exists.

    Uses parameterized INSERT to avoid SQL injection issues.
    """
    cols = [c for c in df.columns if c != 'id']
    placeholders = ','.join(['?'] * (1 + len(cols)))
    col_list = ','.join([f'"id"'] + [f'"{c}"' for c in cols])
    sql = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})'

    to_insert = []
    for _, row in df.iterrows():
        values = [int(row['id'])]
        for c in cols:
            v = row[c]
            if pd.isna(v):
                values.append(None)
            else:
                values.append(v)
        to_insert.append(tuple(values))

    with conn:
        conn.executemany(sql, to_insert)


def main():
    cfg = CONFIG

    csv_path = cfg["csv_path"]
    db_path = cfg["db_path"]

    table = cfg.get("table", "compounds")
    start_id = cfg.get("start_id", 1)
    density_col = cfg.get("density_col", None)

    df = pd.read_csv(csv_path, dtype=str, encoding="gbk")

    # The remaining logic is unchanged

    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    # Find density column
    if not density_col:
        density_col = find_density_column(df.columns)
    if not density_col or density_col not in df.columns:
        print(
            'Warning: could not auto-detect density column. '
            'You can pass --density-col "Column Name"',
            file=sys.stderr
        )
        density_col = None

    out = df.copy()

    # Parse density column
    if density_col:
        out['density_clean'] = out[density_col].apply(parse_density)
    else:
        out['density_clean'] = None

    # Attempt numeric conversion for molecular weight columns
    mw_candidates = [
        c for c in out.columns
        if 'molecular' in c.lower() and 'weight' in c.lower()
    ]
    for c in mw_candidates:
        out[c] = pd.to_numeric(out[c], errors='coerce')

    # Insert id column
    start = start_id if start_id is not None else 1
    out.insert(0, 'id', range(start, start + len(out)))

    # Ensure density_clean is float
    out['density_clean'] = pd.to_numeric(out['density_clean'], errors='coerce')

    conn = sqlite3.connect(db_path)
    try:
        create_table_with_schema(conn, table, out)
        insert_dataframe(conn, table, out)
    finally:
        conn.close()

    print(f'Wrote {len(out)} rows to {db_path} (table: {table}).')


if __name__ == '__main__':
    main()