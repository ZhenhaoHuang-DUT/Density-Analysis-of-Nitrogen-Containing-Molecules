import sqlite3
import pandas as pd
from typing import Union
from pathlib import Path


def explore_database(db_path: Union[str, Path]) -> None:
    """
    Interactively explore the structure and sample content of an SQLite database.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    """

    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    with sqlite3.connect(db_path) as conn:

        # Retrieve all table names
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'",
            conn
        )["name"].tolist()

        print(f"Database: {db_path}")
        print(f"Number of tables found: {len(tables)}")

        for table in tables:
            print("\n" + "*" * 60)
            print(f"Table: {table}")
            print("*" * 60)

            # Retrieve table schema information
            info = pd.read_sql(f"PRAGMA table_info({table})", conn)
            print(f"Number of columns: {len(info)}")
            print("\nColumn information:")
            print(info[["name", "type", "notnull"]].to_string(index=False))

            # Retrieve sample data
            try:
                sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
                if not sample.empty:
                    print("\nFirst 3 rows:")
                    print(sample.to_string(index=False))
                else:
                    print("\n(Empty table)")
            except Exception as e:
                print(f"\nError reading sample data: {e}")


# Example usage
if __name__ == "__main__":
    explore_database("COD_Database.db")
    explore_database("reaxys_N.db")