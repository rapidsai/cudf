# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cast Decimal columns to Float64 in a TPC parquet dataset directory.

Workaround for https://github.com/rapidsai/cudf/issues/23150.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("TPC_DATA_DIR"),
        help="Directory containing parquet files. Defaults to TPC_DATA_DIR environment variable.",
    )
    args = parser.parse_args()

    if args.data_dir is None:
        parser.error("--data-dir is required (or set TPC_DATA_DIR).")

    for path in sorted(Path(args.data_dir).rglob("*.parquet")):
        lf = pl.scan_parquet(path)
        decimal_cols = [
            name
            for name, dtype in lf.collect_schema().items()
            if isinstance(dtype, pl.Decimal)
        ]
        if decimal_cols:
            lf.with_columns(
                pl.col(decimal_cols).cast(pl.Float64)
            ).sink_parquet(path)


if __name__ == "__main__":
    main()
