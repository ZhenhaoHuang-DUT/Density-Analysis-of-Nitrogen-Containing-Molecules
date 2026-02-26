import pandas as pd
import sqlite3
import glob
import os
from typing import Union


def simple_batch_convert(
    csv_folder: Union[str, os.PathLike],
    db_file: str = "../Data/COD_Database.db",
    table_name: str = "CellData"
) -> None:
    """
    Simplified batch conversion of multiple CSV files into a single SQLite table.

    Parameters
    ----------
    csv_folder : str or PathLike
        Directory containing CSV files.
    db_file : str
        Output SQLite database file name.
    table_name : str
        Name of the table to create in the database.
    """

    # Retrieve all CSV files in the folder
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Processing {len(csv_files)} files...")

    # Read and collect all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, sep=",", skiprows=11)
            df = df.rename(columns={"file": "ID"})
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {os.path.basename(csv_file)}: {e}")

    if not dfs:
        print("No files were successfully loaded.")
        return

    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save to SQLite database
    with sqlite3.connect(db_file) as conn:
        combined_df.to_sql(table_name, conn, if_exists="replace", index=False)

        # Create index on ID column
        conn.execute(f"CREATE INDEX idx_id ON {table_name} (ID)")

        # Retrieve statistics
        count = pd.read_sql(
            f"SELECT COUNT(*) as count FROM {table_name}",
            conn
        )["count"][0]

        cols = pd.read_sql(
            f"PRAGMA table_info({table_name})",
            conn
        )

    print("\nCompleted successfully.")
    print(f"Database: {db_file}")
    print(f"Table name: {table_name}")
    print(f"Number of records: {count}")
    print(f"Number of columns: {len(cols)}")
    print(f"\nColumn name preview: {list(cols['name'])[:10]}...")


# Example usage
if __name__ == "__main__":
    simple_batch_convert("../Data/COD_csv")