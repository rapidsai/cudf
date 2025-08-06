#!/usr/bin/env python3
"""
Script to generate a parquet file with 3 columns:
1. Ascending numbers in string format (xxxxxx)
2. Random float data
3. Random integer data
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def generate_parquet_data(num_rows=1000, output_file="generated_data.parquet"):
    """
    Generate a parquet file with 20 columns of various data types.

    Args:
        num_rows (int): Number of rows to generate
        output_file (str): Output parquet file path
    """
    # Generate ascending numbers in string format (xxxxxx) that repeat every 1/4th
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

    # Generate timestamp data (random dates in 2024)
    # Use modulo to avoid overflow for large num_rows
    max_periods = min(num_rows, 10000)  # Limit to prevent overflow
    timestamp_column = pd.date_range(
        "2024-01-01", periods=max_periods, freq="h"
    )
    # Repeat the pattern if needed
    timestamp_column = timestamp_column.tolist() * (
        num_rows // max_periods + 1
    )
    timestamp_column = timestamp_column[:num_rows]

    # Generate boolean data
    bool_column = np.random.choice([True, False], num_rows)

    # Generate decimal data (as strings to maintain precision)
    decimal_column = [
        f"{np.random.uniform(0, 1000):.6f}" for _ in range(num_rows)
    ]

    # Generate list of integers
    list_int_column = [
        [np.random.randint(1, 100) for _ in range(np.random.randint(1, 5))]
        for _ in range(num_rows)
    ]

    # Generate list of doubles
    list_double_column = [
        [np.random.uniform(0, 100) for _ in range(np.random.randint(1, 4))]
        for _ in range(num_rows)
    ]

    # Generate UUID-like strings
    uuid_column = [
        f"uuid-{i:08x}-{np.random.randint(1000, 9999)}-"
        f"{np.random.randint(1000, 9999)}-"
        f"{np.random.randint(100000000000, 999999999999)}"
        for i in range(num_rows)
    ]

    # Generate email addresses
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
    email_column = [
        f"user{i}@{np.random.choice(domains)}" for i in range(num_rows)
    ]

    # Generate IP addresses
    ip_column = [
        f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}."
        f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        for _ in range(num_rows)
    ]

    # Generate time data (time of day)
    # Use modulo to avoid overflow for large num_rows
    max_periods = min(
        num_rows, 86400
    )  # Limit to prevent overflow (seconds in a day)
    time_column = pd.date_range("00:00:00", periods=max_periods, freq="s").time
    # Repeat the pattern if needed
    time_column = time_column.tolist() * (num_rows // max_periods + 1)
    time_column = time_column[:num_rows]

    # Generate date data (just dates)
    # Use modulo to avoid overflow for large num_rows
    max_periods = min(num_rows, 36500)  # Limit to prevent overflow (100 years)
    date_column = pd.date_range(
        "2020-01-01", periods=max_periods, freq="d"
    ).date
    # Repeat the pattern if needed
    date_column = date_column.tolist() * (num_rows // max_periods + 1)
    date_column = date_column[:num_rows]

    # Generate large integers
    large_int_column = np.random.randint(
        1000000000000, 9999999999999, num_rows
    )

    # Generate small floats (precision testing)
    small_float_column = np.random.uniform(0.000001, 0.000999, num_rows)

    # Generate additional simple numeric columns
    short_int_column = np.random.randint(
        -32768, 32767, num_rows
    )  # int16 range
    long_float_column = np.random.uniform(-1e6, 1e6, num_rows)
    positive_int_column = np.random.randint(0, 100000, num_rows)
    negative_float_column = np.random.uniform(-1000, -1, num_rows)

    # Generate new columns with random lists of integers and strings
    # Generate random list of integers column
    random_list_int_column = [
        [
            np.random.randint(-1000, 1000)
            for _ in range(np.random.randint(2, 8))
        ]
        for _ in range(num_rows)
    ]

    # Generate random list of strings column
    words = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
        "orange",
        "papaya",
        "quince",
        "raspberry",
    ]
    random_list_string_column = [
        [random.choice(words) for _ in range(np.random.randint(1, 6))]
        for _ in range(num_rows)
    ]

    # Generate another random list of integers column
    random_list_int_column2 = [
        [np.random.randint(1, 10000) for _ in range(np.random.randint(1, 10))]
        for _ in range(num_rows)
    ]

    # Generate another random list of strings column
    colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "orange",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "teal",
    ]
    random_list_string_column2 = [
        [random.choice(colors) for _ in range(np.random.randint(2, 7))]
        for _ in range(num_rows)
    ]

    # Create the original DataFrame with all columns
    original_df = pd.DataFrame(
        {
            "string_col": string_column,
            "float_col": float_column,
            "int_col": int_column,
            "timestamp_col": timestamp_column,
            "bool_col": bool_column,
            "decimal_col": decimal_column,
            "list_int_col": list_int_column,
            "list_double_col": list_double_column,
            "uuid_col": uuid_column,
            "email_col": email_column,
            "ip_col": ip_column,
            "time_col": time_column,
            "date_col": date_column,
            "large_int_col": large_int_column,
            "small_float_col": small_float_column,
            "short_int_col": short_int_column,
            "long_float_col": long_float_column,
            "positive_int_col": positive_int_column,
            "negative_float_col": negative_float_column,
        }
    )

    # Get all column names except the 0th one (string_col)
    all_columns = list(original_df.columns)
    columns_to_remove = all_columns[1:]  # Exclude the 0th column

    # Randomly select 4 columns to remove
    columns_to_remove = random.sample(columns_to_remove, 4)
    print(f"Removing columns: {columns_to_remove}")

    # Create new DataFrame with removed columns and new list columns
    df = original_df.drop(columns=columns_to_remove)

    # Add the new list columns
    df["random_list_int_col"] = random_list_int_column
    df["random_list_string_col"] = random_list_string_column
    df["random_list_int_col2"] = random_list_int_column2
    df["random_list_string_col2"] = random_list_string_column2

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
    print(f"Number of columns: {len(df.columns)}")

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
        default=1000000,
        help="Number of rows to generate (default: 1000000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_data_20.parquet",
        help="Output parquet file path (default: generated_data_20.parquet)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_parquet_data(args.rows, args.output)


if __name__ == "__main__":
    main()
