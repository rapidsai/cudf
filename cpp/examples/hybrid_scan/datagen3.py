#!/usr/bin/env python3
"""
Script to generate a parquet file with 3 columns:
1. Ascending numbers in string format (xxxxxx)
2. Random float data
3. Random integer data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def generate_parquet_data(num_rows=1000, output_file="generated_data.parquet"):
    """
    Generate a parquet file with the specified columns.

    Args:
        num_rows (int): Number of rows to generate
        output_file (str): Output parquet file path
    """
    # Generate ascending numbers in string format (xxxxxx) that repeat every 1/2th
    # of num_rows
    half_size = num_rows // 2
    base_strings = [f"{i:07d}" for i in range(1, half_size + 1)]
    string_column = base_strings * 2
    # Ensure we have exactly num_rows by trimming or extending
    if len(string_column) < num_rows:
        string_column.extend(base_strings[: num_rows - len(string_column)])
    else:
        string_column = string_column[:num_rows]

    # Generate random float data
    float_column = np.random.uniform(0.0, 1000.0, num_rows)

    # Generate random integer data
    int_column = np.random.randint(1, 1000000, num_rows)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "string_col": string_column,
            "float_col": float_column,
            "int_col": int_column,
        }
    )

    # Convert DataFrame to Arrow Table
    table = pa.Table.from_pandas(df)

    # Write to parquet file with specific row group and data page settings
    pq.write_table(
        table,
        output_file,
        row_group_size=6000,  # 6000 rows per row group
        data_page_size=2000,  # 2000 rows per data page
        write_page_index=True,  # Include page index in the parquet file
        use_dictionary=True,  # Use dictionary encoding for string columns
    )

    print(f"Generated parquet file: {output_file}")
    print(f"Number of rows: {num_rows}")

    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())


def main():
    parser = argparse.ArgumentParser(
        description="Generate a parquet file with sample data"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100000,
        help="Number of rows to generate (default: 100000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_data.parquet",
        help="Output parquet file path (default: generated_data.parquet)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_parquet_data(args.rows, args.output)


if __name__ == "__main__":
    main()
