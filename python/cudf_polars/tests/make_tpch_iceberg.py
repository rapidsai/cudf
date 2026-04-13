#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Convert local TPC-H parquet files to Iceberg format on local disk.

The source layout is a directory of part files per table::

    /datasets/nperera/scale-1000/
    ├── orders/     125 x part.N.parquet
    ├── lineitem/   500 x part.N.parquet
    └── ...

Output Iceberg warehouse::

    /datasets/nperera/scale-1000-iceberg/
    ├── catalog.db          (SQLite catalog)
    ├── orders/             (Iceberg table data + metadata)
    └── ...

Usage::

    # Build only the orders table (needed for the integration test)
    python make_tpch_iceberg.py --tables orders

    # Build all tables
    python make_tpch_iceberg.py

    # Custom paths
    python make_tpch_iceberg.py \\
        --input  /datasets/nperera/scale-1000 \\
        --output /datasets/nperera/scale-1000-iceberg \\
        --tables orders,lineitem
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.io.pyarrow import pyarrow_to_schema, schema_to_pyarrow
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.table.name_mapping import MappedField, NameMapping
from pyiceberg.transforms import MonthTransform

# ── table config ──────────────────────────────────────────────────────────────

TPCH_TABLES = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]

# Tables partitioned by a date column (monthly transform)
DATE_PARTITIONED = {
    "lineitem": "l_shipdate",
    "orders": "o_orderdate",
}

SORT_KEYS: dict[str, list[str]] = {
    "lineitem": ["l_orderkey"],
    "orders": ["o_orderkey"],
    "customer": ["c_custkey"],
    "part": ["p_partkey"],
    "partsupp": ["ps_partkey", "ps_suppkey"],
    "supplier": ["s_suppkey"],
}

NAMESPACE = "tpch"

# ── helpers ───────────────────────────────────────────────────────────────────


def convert_table(
    catalog: SqlCatalog,
    warehouse: str,
    input_base: str,
    table_name: str,
    batch_rows: int = 1_000_000,
) -> int:
    """Convert one TPC-H table directory to an Iceberg table.

    Parameters
    ----------
    catalog
        PyIceberg SqlCatalog to register the table in.
    warehouse
        ``file://`` URI for the Iceberg warehouse root.
    input_base
        Path to the directory containing per-table subdirectories.
        Each table lives at ``{input_base}/{table_name}/``.
    table_name
        TPC-H table name (e.g. ``"orders"``).
    batch_rows
        Rows per write batch (controls memory usage during conversion).

    Returns
    -------
    int
        Total rows written.
    """
    # Source: directory of part files (not a single .parquet file)
    input_path = f"{input_base}/{table_name}"
    print(f"\n{'=' * 60}")
    print(f"[{table_name}] reading from {input_path}")

    fs = pafs.LocalFileSystem()
    dataset = ds.dataset(input_path, filesystem=fs, format="parquet")
    arrow_schema = dataset.schema
    print(f"[{table_name}] {dataset.count_rows():,} rows | {len(arrow_schema)} columns")

    # Assign fresh field IDs via name mapping (source parquet files have none)
    name_mapping = NameMapping(
        root=[
            MappedField(field_id=i + 1, names=[arrow_schema.field(i).name])
            for i in range(len(arrow_schema))
        ]
    )
    iceberg_schema = pyarrow_to_schema(arrow_schema, name_mapping=name_mapping)

    # Build partition spec
    if table_name in DATE_PARTITIONED:
        date_col = DATE_PARTITIONED[table_name]
        field_id = iceberg_schema.find_field(date_col).field_id
        partition_spec = PartitionSpec(
            PartitionField(
                source_id=field_id,
                field_id=1000,
                transform=MonthTransform(),
                name=f"{date_col}_month",
            )
        )
        print(f"[{table_name}] partitioning by months({date_col})")
    else:
        partition_spec = PartitionSpec()
        print(f"[{table_name}] no partitioning (dimension table)")

    # Drop and recreate if exists (idempotent)
    full_name = f"{NAMESPACE}.{table_name}"
    if catalog.table_exists(full_name):
        print(f"[{table_name}] dropping existing Iceberg table")
        catalog.drop_table(full_name)

    ice_table = catalog.create_table(
        identifier=full_name,
        schema=iceberg_schema,
        partition_spec=partition_spec,
        location=f"{warehouse}/{table_name}",
    )
    print(f"[{table_name}] Iceberg table created at {warehouse}/{table_name}")

    arrow_schema_with_ids = schema_to_pyarrow(iceberg_schema)
    sort_cols = SORT_KEYS.get(table_name, [])
    total_rows = 0

    for i, batch in enumerate(dataset.to_batches(batch_size=batch_rows)):
        tbl = pa.Table.from_batches([batch])
        if sort_cols:
            tbl = tbl.sort_by(
                [(c, "ascending") for c in sort_cols if c in tbl.schema.names]
            )
        tbl = pa.Table.from_arrays(
            [
                tbl.column(arrow_schema_with_ids.field(j).name)
                for j in range(len(arrow_schema_with_ids))
            ],
            schema=arrow_schema_with_ids,
        )
        ice_table.append(tbl)
        total_rows += len(tbl)
        print(f"[{table_name}] batch {i + 1}: {total_rows:,} rows written", end="\r")

    print(f"\n[{table_name}] done — {total_rows:,} total rows")
    return total_rows


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Convert local TPC-H parquet directories → Iceberg tables"
    )
    parser.add_argument(
        "--input",
        default="/datasets/nperera/scale-1000",
        help="Directory containing per-table parquet subdirectories",
    )
    parser.add_argument(
        "--output",
        default="/datasets/nperera/scale-1000-iceberg",
        help="Output directory for Iceberg tables (warehouse root)",
    )
    parser.add_argument(
        "--tables",
        default="all",
        help="Comma-separated table names, or 'all'",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="SQLite catalog path (default: <output>/catalog.db)",
    )
    args = parser.parse_args()

    catalog_path = args.catalog or str(Path(args.output) / "catalog.db")
    tables = TPCH_TABLES if args.tables == "all" else args.tables.split(",")

    print(f"Tables to convert: {tables}")
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Catalog: {catalog_path}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    catalog = SqlCatalog(
        NAMESPACE,
        uri=f"sqlite:///{catalog_path}",
        warehouse=f"file://{args.output}",
    )

    if (NAMESPACE,) not in catalog.list_namespaces():
        catalog.create_namespace(NAMESPACE)

    results: dict[str, int] = {}
    failed: list[str] = []
    for table in tables:
        try:
            rows = convert_table(catalog, f"file://{args.output}", args.input, table)
            results[table] = rows
        except Exception as e:
            import traceback

            print(f"\n[{table}] ERROR: {e}", file=sys.stderr)
            traceback.print_exc()
            failed.append(table)

    print(f"\n{'=' * 60}")
    print("Summary:")
    for t, r in results.items():
        print(f"  {t:<12} {r:>15,} rows")
    if failed:
        print(f"\nFailed: {failed}")
    print(f"\nCatalog: {catalog_path}")
    print(f"Data:    {args.output}")


if __name__ == "__main__":
    main()
