import pandas as pd
import sqlite3
from pathlib import Path
from typing import Union


def csv_to_sqlite(
    csv_file: Union[str, Path],
    db_file: Union[str, Path],
    table_name: str
) -> None:
    """
    Convert a CSV file into an SQLite database table.

    Parameters
    ----------
    csv_file : str or Path
        Path to the input CSV file.
    db_file : str or Path
        Path to the output SQLite database file.
    table_name : str
        Name of the table to create in the database.
    """

    csv_file = Path(csv_file)
    db_file = Path(db_file)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    print(f"Reading CSV file: {csv_file}")

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Rename columns (adjust based on actual CSV structure)
    df = df.rename(columns={
        "cod_entries.cod_number": "ID",
        "molecular_entities.smiles": "SMILES"
    })

    # Keep only required columns if they exist
    required_columns = ["ID", "SMILES"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns]

    print(f"Total records loaded: {len(df)}")

    # Connect to SQLite database (creates file if it does not exist)
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        # Write DataFrame to SQLite
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        # Create index on ID column for faster lookup
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_id ON {table_name}(ID);")

        # Verify inserted data
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        print("\nConversion completed successfully.")
        print(f"Database file: {db_file}")
        print(f"Table name: {table_name}")
        print(f"Total records in table: {count}")

        print("\nFirst 5 records:")
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        rows = cursor.fetchall()

        for row in rows:
            print(f"  ID: {row[0]}, SMILES: {row[1]}")


if __name__ == "__main__":
    # Example usage
    csv_path = Path("../Data/CODid_SMILES.csv")
    database_path = Path("../Data/COD_Database.db")
    table = "SMILES"

    csv_to_sqlite(csv_path, database_path, table)