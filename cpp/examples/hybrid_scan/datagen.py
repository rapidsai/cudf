# Copyright (c) 2019-2025, NVIDIA CORPORATION.

"""
AI generated script to generate a parquet file with 50 columns of various
data types such as strings, numbers, timestamps, booleans, lists, etc.
The first column in the generated data is called `string_col` and contains
strings of 7-digit ascending numbers like "0000000", "0000001" for half
the rows and then repeats itself
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def add_nulls_to_column(column_data, null_probability=0.25):
    """Add nulls to a column at the specified percentage."""
    if null_probability <= 0:
        return column_data

    num_nulls = int(len(column_data) * null_probability)
    null_indices = random.sample(range(len(column_data)), num_nulls)

    if isinstance(column_data, np.ndarray):
        column_data = column_data.tolist()

    for idx in null_indices:
        column_data[idx] = None

    return column_data


def add_nulls_to_list_column(list_column_data, null_probability=0.25):
    """Add nulls to a list column at all depths."""
    if null_probability <= 0:
        return list_column_data

    if isinstance(list_column_data, np.ndarray):
        list_column_data = list_column_data.tolist()

    # Add nulls to entire lists
    if null_probability > 0:
        num_list_nulls = int(len(list_column_data) * null_probability)
        list_null_indices = random.sample(
            range(len(list_column_data)), num_list_nulls
        )
        for idx in list_null_indices:
            list_column_data[idx] = None

    # Add nulls to elements within lists
    if null_probability > 0:
        for item in list_column_data:
            if item is not None and isinstance(item, list):
                num_element_nulls = int(len(item) * null_probability)
                if num_element_nulls > 0:
                    element_null_indices = random.sample(
                        range(len(item)), num_element_nulls
                    )
                    for idx in element_null_indices:
                        item[idx] = None

    return list_column_data


def create_nested_list(depth=0, max_depth=3, data_type="int"):
    """Create nested lists of various data types."""
    # Create a random generator for this function
    rng = np.random.default_rng()

    if depth >= max_depth:
        if data_type == "int":
            return rng.integers(-1000, 1000)
        elif data_type == "float":
            return rng.uniform(-100.0, 100.0)
        elif data_type == "string":
            fruits = [
                "apple",
                "banana",
                "cherry",
                "date",
                "elderberry",
                "fig",
                "grape",
            ]
            return random.choice(fruits)
        elif data_type == "bool":
            return random.choice([True, False])
        elif data_type == "timestamp":
            return pd.Timestamp.now() + pd.Timedelta(
                days=rng.integers(-365, 365)
            )
        elif data_type == "date":
            base_date = pd.date_range("2020-01-01", periods=1, freq="D")[0]
            return base_date + pd.Timedelta(days=rng.integers(-1000, 1000))
        elif data_type == "time":
            return (
                pd.Timestamp("00:00:00")
                + pd.Timedelta(hours=rng.integers(0, 24))
            ).time()
        else:  # mixed
            return str(random.choice([rng.integers(1, 100), "mixed"]))

    if depth == 0:
        return [
            create_nested_list(depth + 1, max_depth, data_type)
            for _ in range(rng.integers(2, 5))
        ]
    elif depth == 1:
        return [
            create_nested_list(depth + 1, max_depth, data_type)
            for _ in range(rng.integers(1, 4))
        ]
    else:
        return [
            create_nested_list(depth + 1, max_depth, data_type)
            for _ in range(rng.integers(1, 3))
        ]


def generate_column_group(group_data, num_rows):
    """Generate a group of columns in parallel."""
    base_range = range(num_rows)
    columns = {}

    # Create a random generator for this thread
    rng = np.random.default_rng()

    for col_name, col_type, col_params in group_data:
        if col_type == "string":
            if col_name == "string_col":
                base_strings = [
                    f"{i:07d}"
                    for i in range(1, min(num_rows // 2 + 1, num_rows + 1))
                ]
                columns[col_name] = (
                    base_strings * (num_rows // len(base_strings) + 1)
                )[:num_rows]
            elif col_name == "string_col_2":
                columns[col_name] = [f"str_{i:08d}" for i in base_range]
            elif col_name == "string_col_3":
                columns[col_name] = [
                    f"text_{rng.integers(1000, 9999)}" for _ in base_range
                ]
            elif col_name == "string_col_4":
                columns[col_name] = [
                    f"label_{chr(65 + i % 26)}{i % 100}" for i in base_range
                ]
            elif col_name == "string_col_5":
                columns[col_name] = [
                    f"code_{rng.integers(100000, 999999)}" for _ in base_range
                ]
            elif col_name == "uuid_col":
                columns[col_name] = [
                    f"uuid-{i:08x}-{rng.integers(1000, 9999)}-{rng.integers(1000, 9999)}-{rng.integers(100000000000, 999999999999)}"
                    for i in base_range
                ]
            elif col_name == "email_col":
                columns[col_name] = [
                    f"user{i}@{random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'])}"
                    for i in base_range
                ]
            elif col_name == "ip_col":
                columns[col_name] = [
                    f"{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}"
                    for _ in base_range
                ]
            elif col_name == "decimal_col":
                columns[col_name] = [
                    f"{rng.uniform(0, 1000):.6f}" for _ in base_range
                ]

        elif col_type == "float":
            if col_name == "float_col":
                columns[col_name] = rng.uniform(0.0, 1000.0, num_rows)
            elif col_name == "float_col_2":
                columns[col_name] = rng.uniform(-500.0, 500.0, num_rows)
            elif col_name == "float_col_3":
                columns[col_name] = rng.uniform(0.0, 1.0, num_rows)
            elif col_name == "float_col_4":
                columns[col_name] = rng.uniform(10000.0, 50000.0, num_rows)
            elif col_name == "float_col_5":
                columns[col_name] = rng.uniform(-0.1, 0.1, num_rows)
            elif col_name == "small_float_col":
                columns[col_name] = rng.uniform(0.000001, 0.000999, num_rows)
            elif col_name == "long_float_column":
                columns[col_name] = rng.uniform(-1e6, 1e6, num_rows)
            elif col_name == "negative_float_column":
                columns[col_name] = rng.uniform(-1000, -1, num_rows)

        elif col_type == "int":
            if col_name == "int_col":
                columns[col_name] = rng.integers(1, 1000000, num_rows)
            elif col_name == "int_col_2":
                columns[col_name] = rng.integers(-10000, 10000, num_rows)
            elif col_name == "int_col_3":
                columns[col_name] = rng.integers(100, 999, num_rows)
            elif col_name == "int_col_4":
                columns[col_name] = rng.integers(1000000, 9999999, num_rows)
            elif col_name == "int_col_5":
                columns[col_name] = rng.integers(0, 255, num_rows)
            elif col_name == "short_int_column":
                columns[col_name] = rng.integers(-32768, 32767, num_rows)
            elif col_name == "positive_int_column":
                columns[col_name] = rng.integers(0, 100000, num_rows)
            elif col_name == "large_int_column":
                columns[col_name] = rng.integers(
                    1000000000000, 9999999999999, num_rows
                )

        elif col_type == "bool":
            if col_name == "bool_col":
                columns[col_name] = rng.choice([True, False], num_rows)
            elif col_name == "bool_col_2":
                columns[col_name] = rng.choice(
                    [True, False], num_rows, p=[0.3, 0.7]
                )
            elif col_name == "bool_col_3":
                columns[col_name] = rng.choice(
                    [True, False], num_rows, p=[0.7, 0.3]
                )
            elif col_name == "bool_col_4":
                columns[col_name] = [i % 2 == 0 for i in base_range]
            elif col_name == "bool_col_5":
                columns[col_name] = [i % 3 == 0 for i in base_range]

        elif col_type == "timestamp":
            if col_name == "timestamp_col":
                max_periods = min(num_rows, 10000)
                base_timestamps = pd.date_range(
                    "2024-01-01", periods=max_periods, freq="h"
                )
                columns[col_name] = (
                    base_timestamps.tolist() * (num_rows // max_periods + 1)
                )[:num_rows]
            elif col_name == "timestamp_col_2":
                columns[col_name] = (
                    pd.date_range(
                        "2023-01-01", periods=min(num_rows, 10000), freq="D"
                    ).tolist()
                    * (num_rows // min(num_rows, 10000) + 1)
                )[:num_rows]
            elif col_name == "timestamp_col_3":
                columns[col_name] = (
                    pd.date_range(
                        "2025-01-01", periods=min(num_rows, 1000), freq="W"
                    ).tolist()
                    * (num_rows // min(num_rows, 1000) + 1)
                )[:num_rows]

        elif col_type == "date":
            if col_name == "date_col":
                max_periods = min(num_rows, 36500)
                base_dates = pd.date_range(
                    "2020-01-01", periods=max_periods, freq="d"
                ).date
                columns[col_name] = (
                    base_dates.tolist() * (num_rows // max_periods + 1)
                )[:num_rows]
            elif col_name == "date_col_2":
                columns[col_name] = (
                    pd.date_range(
                        "2019-01-01", periods=min(num_rows, 3650), freq="W"
                    ).date.tolist()
                    * (num_rows // min(num_rows, 3650) + 1)
                )[:num_rows]

        elif col_type == "time":
            if col_name == "time_col":
                max_periods = min(num_rows, 1440)
                base_times = pd.date_range(
                    "00:00:00", periods=max_periods, freq="min"
                ).time
                columns[col_name] = (
                    base_times.tolist() * (num_rows // max_periods + 1)
                )[:num_rows]
            elif col_name == "time_col_2":
                columns[col_name] = (
                    pd.date_range(
                        "12:00:00", periods=min(num_rows, 1440), freq="min"
                    ).time.tolist()
                    * (num_rows // min(num_rows, 1440) + 1)
                )[:num_rows]

        elif col_type == "list":
            columns[col_name] = [
                create_nested_list(
                    0, col_params["depth"], col_params["data_type"]
                )
                for _ in base_range
            ]

    return columns


def generate_parquet_data(
    num_rows=1000,
    output_file="generated_data_50.parquet",
    null_probability=0.1,
    row_group_size=6000,
    data_page_size=2000,
):
    """Generate a parquet file with 50 columns of various data types using multiple threads."""
    start_time = time.time()

    # Create a random generator for this function
    rng = np.random.default_rng()

    # Base data that can be reused
    base_range = range(num_rows)
    base_strings = [
        f"{i:07d}" for i in range(1, min(num_rows // 2 + 1, num_rows + 1))
    ]
    string_column = (base_strings * (num_rows // len(base_strings) + 1))[
        :num_rows
    ]

    # Generate all data at once
    float_data = rng.uniform(0.0, 1000.0, num_rows)
    int_data = rng.integers(1, 1000000, num_rows)
    bool_data = rng.choice([True, False], num_rows)

    # Reuse random data for multiple columns
    rand_ints = rng.integers(-10000, 10000, num_rows)
    rand_floats = rng.uniform(-500.0, 500.0, num_rows)

    # Generate timestamps, dates, and times efficiently
    max_periods = min(num_rows, 10000)
    base_timestamps = pd.date_range(
        "2024-01-01", periods=max_periods, freq="h"
    )
    timestamp_column = (
        base_timestamps.tolist() * (num_rows // max_periods + 1)
    )[:num_rows]

    max_periods = min(num_rows, 36500)
    base_dates = pd.date_range(
        "2020-01-01", periods=max_periods, freq="d"
    ).date
    date_column = (base_dates.tolist() * (num_rows // max_periods + 1))[
        :num_rows
    ]

    max_periods = min(num_rows, 1440)
    base_times = pd.date_range(
        "00:00:00", periods=max_periods, freq="min"
    ).time
    time_column = (base_times.tolist() * (num_rows // max_periods + 1))[
        :num_rows
    ]

    # Create DataFrame with all columns
    df = pd.DataFrame(
        {
            # Core columns (reusing base data)
            "string_col": string_column,
            "float_col": float_data,
            "int_col": int_data,
            "timestamp_col": timestamp_column,
            "bool_col": bool_data,
            "date_col": date_column,
            "time_col": time_column,
            # String variations
            "string_col_2": [f"str_{i:08d}" for i in base_range],
            "string_col_3": [
                f"text_{rng.integers(1000, 9999)}" for _ in base_range
            ],
            "string_col_4": [
                f"label_{chr(65 + i % 26)}{i % 100}" for i in base_range
            ],
            "string_col_5": [
                f"code_{rng.integers(100000, 999999)}" for _ in base_range
            ],
            # Float variations (reusing random data)
            "float_col_2": rand_floats,
            "float_col_3": rng.uniform(0.0, 1.0, num_rows),
            "float_col_4": rng.uniform(10000.0, 50000.0, num_rows),
            "float_col_5": rng.uniform(-0.1, 0.1, num_rows),
            "small_float_col": rng.uniform(0.000001, 0.000999, num_rows),
            "long_float_column": rng.uniform(-1e6, 1e6, num_rows),
            "negative_float_column": rng.uniform(-1000, -1, num_rows),
            # Integer variations (reusing random data)
            "int_col_2": rand_ints,
            "int_col_3": rng.integers(100, 999, num_rows),
            "int_col_4": rng.integers(1000000, 9999999, num_rows),
            "int_col_5": rng.integers(0, 255, num_rows),
            "short_int_column": rng.integers(-32768, 32767, num_rows),
            "positive_int_column": rng.integers(0, 100000, num_rows),
            "large_int_column": rng.integers(
                1000000000000, 9999999999999, num_rows
            ),
            # Boolean variations (reusing base data)
            "bool_col_2": rng.choice([True, False], num_rows, p=[0.3, 0.7]),
            "bool_col_3": rng.choice([True, False], num_rows, p=[0.7, 0.3]),
            "bool_col_4": [i % 2 == 0 for i in base_range],
            "bool_col_5": [i % 3 == 0 for i in base_range],
            # Timestamp variations
            "timestamp_col_2": (
                pd.date_range(
                    "2023-01-01", periods=min(num_rows, 10000), freq="D"
                ).tolist()
                * (num_rows // min(num_rows, 10000) + 1)
            )[:num_rows],
            "timestamp_col_3": (
                pd.date_range(
                    "2025-01-01", periods=min(num_rows, 1000), freq="W"
                ).tolist()
                * (num_rows // min(num_rows, 1000) + 1)
            )[:num_rows],
            # Time variations
            "time_col_2": (
                pd.date_range(
                    "12:00:00", periods=min(num_rows, 1440), freq="min"
                ).time.tolist()
                * (num_rows // min(num_rows, 1440) + 1)
            )[:num_rows],
            # Date variations
            "date_col_2": (
                pd.date_range(
                    "2019-01-01", periods=min(num_rows, 3650), freq="W"
                ).date.tolist()
                * (num_rows // min(num_rows, 3650) + 1)
            )[:num_rows],
            # Decimal columns
            "decimal_col": [f"{rng.uniform(0, 1000):.6f}" for _ in base_range],
            # UUID and other string columns
            "uuid_col": [
                f"uuid-{i:08x}-{rng.integers(1000, 9999)}-{rng.integers(1000, 9999)}-{rng.integers(100000000000, 999999999999)}"
                for i in base_range
            ],
            "email_col": [
                f"user{i}@{random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'])}"
                for i in base_range
            ],
            "ip_col": [
                f"{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}"
                for _ in base_range
            ],
            # List columns (13 total)
            "list_int_col_1": [
                create_nested_list(0, 3, "int") for _ in base_range
            ],
            "list_float_col_1": [
                create_nested_list(0, 3, "float") for _ in base_range
            ],
            "list_string_col_1": [
                create_nested_list(0, 3, "string") for _ in base_range
            ],
            "list_timestamp_col_1": [
                create_nested_list(0, 3, "timestamp") for _ in base_range
            ],
            "list_bool_col_1": [
                create_nested_list(0, 3, "bool") for _ in base_range
            ],
            "list_date_col_1": [
                create_nested_list(0, 3, "date") for _ in base_range
            ],
            "list_time_col_1": [
                create_nested_list(0, 3, "time") for _ in base_range
            ],
            "list_mixed_col_1": [
                create_nested_list(0, 3, "mixed") for _ in base_range
            ],
            "list_int_col_2": [
                create_nested_list(0, 2, "int") for _ in base_range
            ],
            "list_float_col_2": [
                create_nested_list(0, 2, "float") for _ in base_range
            ],
            "list_string_col_2": [
                create_nested_list(0, 2, "string") for _ in base_range
            ],
            "list_timestamp_col_2": [
                create_nested_list(0, 2, "timestamp") for _ in base_range
            ],
            "list_bool_col_2": [
                create_nested_list(0, 2, "bool") for _ in base_range
            ],
        }
    )

    # Add nulls to all columns
    for col in df.columns:
        if col.startswith("list_"):
            df[col] = add_nulls_to_list_column(
                df[col].tolist(), null_probability
            )
        else:
            df[col] = add_nulls_to_column(df[col].tolist(), null_probability)

    # Convert DataFrame to Arrow Table and write to parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(
        table,
        output_file,
        row_group_size=row_group_size,
        data_page_size=data_page_size,
        write_page_index=True,
        use_dictionary=True,
    )

    generation_time = time.time() - start_time
    print(f"Generated parquet file: {output_file}")
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Column generation time: {generation_time:.2f} seconds")
    print("\nFirst 5 rows:")
    print(df.head())


def main():
    parser = argparse.ArgumentParser(
        description="Generate a parquet file with 50 columns of sample data"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Number of rows to generate (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_data_50.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--nulls",
        type=float,
        default=0.25,
        help="Probability of nulls in columns (0.0 to 1.0, default: 0.25)",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=200,
        help="Number of rows per row group for parquet file (default: 200)",
    )
    parser.add_argument(
        "--data-page-size",
        type=int,
        default=100,
        help="Number of rows per data page for parquet file (default: 100)",
    )

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate_parquet_data(
        args.rows,
        args.output,
        args.nulls,
        args.row_group_size,
        args.data_page_size,
    )


if __name__ == "__main__":
    main()
