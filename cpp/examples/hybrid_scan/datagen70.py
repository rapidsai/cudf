#!/usr/bin/env python3
"""
Script to generate a parquet file with 50 columns:
Various data types including strings, numbers, timestamps, booleans, etc.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def generate_parquet_data(
    num_rows=1000, output_file="generated_data_50.parquet"
):
    """
    Generate a parquet file with 50 columns of various data types.

    Args:
        num_rows (int): Number of rows to generate
        output_file (str): Output parquet file path
    """
    # Generate ascending numbers in string format (xxxxxx) that repeat every
    # 1/2th of num_rows
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

    # Generate additional columns with variations of existing types
    # Additional string columns
    string_col_2 = [f"str_{i:08d}" for i in range(num_rows)]
    string_col_3 = [
        f"text_{np.random.randint(1000, 9999)}" for _ in range(num_rows)
    ]
    string_col_4 = [
        f"label_{chr(65 + i % 26)}{i % 100}" for i in range(num_rows)
    ]
    string_col_5 = [
        f"code_{np.random.randint(100000, 999999)}" for _ in range(num_rows)
    ]

    # Additional float columns
    float_col_2 = np.random.uniform(-500.0, 500.0, num_rows)
    float_col_3 = np.random.uniform(0.0, 1.0, num_rows)
    float_col_4 = np.random.uniform(10000.0, 50000.0, num_rows)
    float_col_5 = np.random.uniform(-0.1, 0.1, num_rows)

    # Additional integer columns
    int_col_2 = np.random.randint(-10000, 10000, num_rows)
    int_col_3 = np.random.randint(100, 999, num_rows)
    int_col_4 = np.random.randint(1000000, 9999999, num_rows)
    int_col_5 = np.random.randint(0, 255, num_rows)

    # Additional timestamp columns
    timestamp_col_2 = pd.date_range(
        "2023-01-01", periods=min(num_rows, 10000), freq="D"
    )
    timestamp_col_2 = timestamp_col_2.tolist() * (
        num_rows // min(num_rows, 10000) + 1
    )
    timestamp_col_2 = timestamp_col_2[:num_rows]

    timestamp_col_3 = pd.date_range(
        "2025-01-01", periods=min(num_rows, 10000), freq="W"
    )
    timestamp_col_3 = timestamp_col_3.tolist() * (
        num_rows // min(num_rows, 10000) + 1
    )
    timestamp_col_3 = timestamp_col_3[:num_rows]

    # Additional boolean columns
    bool_col_2 = np.random.choice([True, False], num_rows, p=[0.3, 0.7])
    bool_col_3 = np.random.choice([True, False], num_rows, p=[0.7, 0.3])
    bool_col_4 = [i % 2 == 0 for i in range(num_rows)]
    bool_col_5 = [i % 3 == 0 for i in range(num_rows)]

    # Additional decimal columns
    decimal_col_2 = [
        f"{np.random.uniform(-100, 100):.4f}" for _ in range(num_rows)
    ]
    decimal_col_3 = [f"{np.random.uniform(0, 1):.8f}" for _ in range(num_rows)]
    decimal_col_4 = [
        f"{np.random.uniform(1000, 9999):.2f}" for _ in range(num_rows)
    ]
    decimal_col_5 = [
        f"{np.random.uniform(-0.001, 0.001):.6f}" for _ in range(num_rows)
    ]

    # Additional UUID columns
    uuid_col_2 = [
        f"id-{i:06x}-{np.random.randint(100, 999)}-"
        f"{np.random.randint(100, 999)}-"
        f"{np.random.randint(10000000, 99999999)}"
        for i in range(num_rows)
    ]
    uuid_col_3 = [
        f"ref-{np.random.randint(100000, 999999)}-"
        f"{np.random.randint(1000, 9999)}-"
        f"{np.random.randint(100000000, 999999999)}"
        for _ in range(num_rows)
    ]

    # Additional email columns
    domains_2 = ["company.com", "org.net", "test.io"]
    email_col_2 = [
        f"test{i}@{np.random.choice(domains_2)}" for i in range(num_rows)
    ]
    email_col_3 = [
        f"admin{np.random.randint(1, 1000)}@{np.random.choice(domains_2)}"
        for _ in range(num_rows)
    ]

    # Additional IP columns
    ip_col_2 = [
        f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        for _ in range(num_rows)
    ]
    ip_col_3 = [
        f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        for _ in range(num_rows)
    ]

    # Additional time columns
    time_col_2 = pd.date_range(
        "12:00:00", periods=min(num_rows, 1440), freq="min"
    ).time
    time_col_2 = time_col_2.tolist() * (num_rows // min(num_rows, 1440) + 1)
    time_col_2 = time_col_2[:num_rows]

    # Additional date columns
    date_col_2 = pd.date_range(
        "2019-01-01", periods=min(num_rows, 3650), freq="W"
    ).date
    date_col_2 = date_col_2.tolist() * (num_rows // min(num_rows, 3650) + 1)
    date_col_2 = date_col_2[:num_rows]

    # Additional numeric columns
    large_int_col_2 = np.random.randint(100000000000, 999999999999, num_rows)
    large_int_col_3 = np.random.randint(-999999999999, -100000000000, num_rows)
    small_float_col_2 = np.random.uniform(0.0000001, 0.0000999, num_rows)
    small_float_col_3 = np.random.uniform(-0.0000999, -0.0000001, num_rows)
    short_int_col_2 = np.random.randint(-10000, 10000, num_rows)
    short_int_col_3 = np.random.randint(0, 65535, num_rows)
    long_float_col_2 = np.random.uniform(-1e8, 1e8, num_rows)
    long_float_col_3 = np.random.uniform(-1e4, 1e4, num_rows)
    positive_int_col_2 = np.random.randint(1, 1000, num_rows)
    positive_int_col_3 = np.random.randint(1000000, 9999999, num_rows)
    negative_float_col_2 = np.random.uniform(-100, -10, num_rows)
    negative_float_col_3 = np.random.uniform(-0.1, -0.01, num_rows)

    # Additional string columns to replace list columns
    string_col_6 = [f"item_{i:06d}" for i in range(num_rows)]
    string_col_7 = [
        f"tag_{np.random.randint(100, 999)}" for _ in range(num_rows)
    ]
    string_col_8 = [
        f"name_{chr(97 + i % 26)}{i % 1000}" for i in range(num_rows)
    ]
    string_col_9 = [
        f"key_{np.random.randint(10000, 99999)}" for _ in range(num_rows)
    ]
    string_col_10 = [
        f"val_{np.random.randint(1000000, 9999999)}" for _ in range(num_rows)
    ]

    # Additional float columns to replace list columns
    float_col_6 = np.random.uniform(-1000.0, 1000.0, num_rows)
    float_col_7 = np.random.uniform(0.0, 10.0, num_rows)
    float_col_8 = np.random.uniform(-100.0, 100.0, num_rows)
    float_col_9 = np.random.uniform(1.0, 1000.0, num_rows)
    float_col_10 = np.random.uniform(-0.01, 0.01, num_rows)

    # Additional integer columns to replace list columns
    int_col_6 = np.random.randint(-5000, 5000, num_rows)
    int_col_7 = np.random.randint(50, 500, num_rows)
    int_col_8 = np.random.randint(500000, 5000000, num_rows)
    int_col_9 = np.random.randint(0, 100, num_rows)
    int_col_10 = np.random.randint(-1000, 1000, num_rows)

    # Additional boolean columns to replace list columns
    bool_col_6 = np.random.choice([True, False], num_rows, p=[0.5, 0.5])
    bool_col_7 = np.random.choice([True, False], num_rows, p=[0.2, 0.8])
    bool_col_8 = np.random.choice([True, False], num_rows, p=[0.8, 0.2])
    bool_col_9 = [i % 4 == 0 for i in range(num_rows)]
    bool_col_10 = [i % 5 == 0 for i in range(num_rows)]

    # Additional timestamp columns to replace list columns
    timestamp_col_4 = pd.date_range(
        "2022-01-01", periods=min(num_rows, 1000), freq="D"
    )
    timestamp_col_4 = timestamp_col_4.tolist() * (
        num_rows // min(num_rows, 1000) + 1
    )
    timestamp_col_4 = timestamp_col_4[:num_rows]

    timestamp_col_5 = pd.date_range(
        "2026-01-01", periods=min(num_rows, 1000), freq="W"
    )
    timestamp_col_5 = timestamp_col_5.tolist() * (
        num_rows // min(num_rows, 1000) + 1
    )
    timestamp_col_5 = timestamp_col_5[:num_rows]

    # Additional decimal columns to replace list columns
    decimal_col_6 = [
        f"{np.random.uniform(-50, 50):.3f}" for _ in range(num_rows)
    ]
    decimal_col_7 = [
        f"{np.random.uniform(0, 10):.5f}" for _ in range(num_rows)
    ]
    decimal_col_8 = [
        f"{np.random.uniform(500, 5000):.1f}" for _ in range(num_rows)
    ]
    decimal_col_9 = [
        f"{np.random.uniform(-0.01, 0.01):.7f}" for _ in range(num_rows)
    ]
    decimal_col_10 = [
        f"{np.random.uniform(100, 1000):.4f}" for _ in range(num_rows)
    ]

    # Generate new list columns to replace removed columns
    # List of integers
    list_int_col_1 = [
        [np.random.randint(-1000, 1000)
         for _ in range(np.random.randint(2, 8))]
        for _ in range(num_rows)
    ]
    list_int_col_2 = [
        [np.random.randint(1, 10000) for _ in range(np.random.randint(1, 10))]
        for _ in range(num_rows)
    ]
    list_int_col_3 = [
        [np.random.randint(-500, 500) for _ in range(np.random.randint(3, 12))]
        for _ in range(num_rows)
    ]

    # List of floats
    list_float_col_1 = [
        [np.random.uniform(-100.0, 100.0)
         for _ in range(np.random.randint(2, 6))]
        for _ in range(num_rows)
    ]
    list_float_col_2 = [
        [np.random.uniform(0.0, 1.0) for _ in range(np.random.randint(1, 8))]
        for _ in range(num_rows)
    ]
    list_float_col_3 = [
        [np.random.uniform(-10.0, 10.0)
         for _ in range(np.random.randint(2, 9))]
        for _ in range(num_rows)
    ]

    # List of strings
    fruits = [
        "apple", "banana", "cherry", "date", "elderberry", "fig", "grape",
        "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange",
        "papaya", "quince", "raspberry"
    ]
    list_string_col_1 = [
        [random.choice(fruits) for _ in range(np.random.randint(1, 6))]
        for _ in range(num_rows)
    ]

    colors = [
        "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown",
        "black", "white", "gray", "cyan", "magenta", "lime", "navy", "teal"
    ]
    list_string_col_2 = [
        [random.choice(colors) for _ in range(np.random.randint(2, 7))]
        for _ in range(num_rows)
    ]

    animals = [
        "cat", "dog", "bird", "fish", "horse", "cow", "pig", "sheep",
        "goat", "rabbit", "hamster", "guinea_pig", "ferret", "chinchilla"
    ]
    list_string_col_3 = [
        [random.choice(animals) for _ in range(np.random.randint(1, 5))]
        for _ in range(num_rows)
    ]

    # List of timestamps
    list_timestamp_col_1 = [
        [pd.Timestamp.now() + pd.Timedelta(days=np.random.randint(-365, 365))
         for _ in range(np.random.randint(1, 4))]
        for _ in range(num_rows)
    ]
    list_timestamp_col_2 = [
        [pd.Timestamp("2024-01-01") +
         pd.Timedelta(hours=np.random.randint(0, 8760))
         for _ in range(np.random.randint(2, 6))]
        for _ in range(num_rows)
    ]

    # List of booleans
    list_bool_col_1 = [
        [random.choice([True, False]) for _ in range(np.random.randint(2, 8))]
        for _ in range(num_rows)
    ]
    list_bool_col_2 = [
        [i % 2 == 0 for i in range(np.random.randint(3, 10))]
        for _ in range(num_rows)
    ]

    # List of dates
    list_date_col_1 = [
        [pd.date_range("2020-01-01", periods=1, freq="D")[0] +
         pd.Timedelta(days=np.random.randint(-1000, 1000))
         for _ in range(np.random.randint(1, 5))]
        for _ in range(num_rows)
    ]

    # List of times
    list_time_col_1 = [
        [(pd.Timestamp("00:00:00") +
          pd.Timedelta(hours=np.random.randint(0, 24))).time()
         for _ in range(np.random.randint(2, 6))]
        for _ in range(num_rows)
    ]

    # List of mixed types (integers and strings)
    list_mixed_col_1 = [
        [str(random.choice([np.random.randint(1, 100), random.choice(fruits)]))
         for _ in range(np.random.randint(2, 7))]
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
            "string_col_2": string_col_2,
            "string_col_3": string_col_3,
            "string_col_4": string_col_4,
            "string_col_5": string_col_5,
            "float_col_2": float_col_2,
            "float_col_3": float_col_3,
            "float_col_4": float_col_4,
            "float_col_5": float_col_5,
            "int_col_2": int_col_2,
            "int_col_3": int_col_3,
            "int_col_4": int_col_4,
            "int_col_5": int_col_5,
            "timestamp_col_2": timestamp_col_2,
            "timestamp_col_3": timestamp_col_3,
            "bool_col_2": bool_col_2,
            "bool_col_3": bool_col_3,
            "bool_col_4": bool_col_4,
            "bool_col_5": bool_col_5,
            "decimal_col_2": decimal_col_2,
            "decimal_col_3": decimal_col_3,
            "decimal_col_4": decimal_col_4,
            "decimal_col_5": decimal_col_5,
            "uuid_col_2": uuid_col_2,
            "uuid_col_3": uuid_col_3,
            "email_col_2": email_col_2,
            "email_col_3": email_col_3,
            "ip_col_2": ip_col_2,
            "ip_col_3": ip_col_3,
            "time_col_2": time_col_2,
            "date_col_2": date_col_2,
            "large_int_col_2": large_int_col_2,
            "large_int_col_3": large_int_col_3,
            "small_float_col_2": small_float_col_2,
            "small_float_col_3": small_float_col_3,
            "short_int_col_2": short_int_col_2,
            "short_int_col_3": short_int_col_3,
            "long_float_col_2": long_float_col_2,
            "long_float_col_3": long_float_col_3,
            "positive_int_col_2": positive_int_col_2,
            "positive_int_col_3": positive_int_col_3,
            "negative_float_col_2": negative_float_col_2,
            "negative_float_col_3": negative_float_col_3,
            "string_col_6": string_col_6,
            "string_col_7": string_col_7,
            "string_col_8": string_col_8,
            "string_col_9": string_col_9,
            "string_col_10": string_col_10,
            "float_col_6": float_col_6,
            "float_col_7": float_col_7,
            "float_col_8": float_col_8,
            "float_col_9": float_col_9,
            "float_col_10": float_col_10,
            "int_col_6": int_col_6,
            "int_col_7": int_col_7,
            "int_col_8": int_col_8,
            "int_col_9": int_col_9,
            "int_col_10": int_col_10,
            "bool_col_6": bool_col_6,
            "bool_col_7": bool_col_7,
            "bool_col_8": bool_col_8,
            "bool_col_9": bool_col_9,
            "bool_col_10": bool_col_10,
            "timestamp_col_4": timestamp_col_4,
            "timestamp_col_5": timestamp_col_5,
            "decimal_col_6": decimal_col_6,
            "decimal_col_7": decimal_col_7,
            "decimal_col_8": decimal_col_8,
            "decimal_col_9": decimal_col_9,
            "decimal_col_10": decimal_col_10,
        }
    )

    # Get all column names except the 0th one (string_col)
    all_columns = list(original_df.columns)
    columns_to_remove = all_columns[1:]  # Exclude the 0th column

    # Randomly select 15 columns to remove
    columns_to_remove = random.sample(columns_to_remove, 15)

    # Create new DataFrame with removed columns and new list columns
    df = original_df.drop(columns=columns_to_remove)

    # Add the new list columns
    df["list_int_col_1"] = list_int_col_1
    df["list_int_col_2"] = list_int_col_2
    df["list_int_col_3"] = list_int_col_3
    df["list_float_col_1"] = list_float_col_1
    df["list_float_col_2"] = list_float_col_2
    df["list_float_col_3"] = list_float_col_3
    df["list_string_col_1"] = list_string_col_1
    df["list_string_col_2"] = list_string_col_2
    df["list_string_col_3"] = list_string_col_3
    df["list_timestamp_col_1"] = list_timestamp_col_1
    df["list_timestamp_col_2"] = list_timestamp_col_2
    df["list_bool_col_1"] = list_bool_col_1
    df["list_bool_col_2"] = list_bool_col_2
    df["list_date_col_1"] = list_date_col_1
    df["list_time_col_1"] = list_time_col_1
    df["list_mixed_col_1"] = list_mixed_col_1

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
        description="Generate a parquet file with 50 columns of sample data"
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
        default="generated_data_70.parquet",
        help="Output parquet file path (default: generated_data_70.parquet)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_parquet_data(args.rows, args.output)


if __name__ == "__main__":
    main()
