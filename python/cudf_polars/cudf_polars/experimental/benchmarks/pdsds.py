# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Experimental PDS-DS benchmarks.

Based on https://github.com/pola-rs/polars-benchmark.

WARNING: This is an experimental (and unofficial)
benchmark script. It is not intended for public use
and may be modified or removed at any time.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    from cudf_polars.experimental.benchmarks.utils import (
        run_duckdb,
        run_polars,
        run_validate,
    )

if TYPE_CHECKING:
    from types import ModuleType

# Without this setting, the first IO task to run
# on each worker takes ~15 sec extra
os.environ["KVIKIO_COMPAT_MODE"] = os.environ.get("KVIKIO_COMPAT_MODE", "on")
os.environ["KVIKIO_NTHREADS"] = os.environ.get("KVIKIO_NTHREADS", "8")


def valid_query(name: str) -> bool:
    """Return True for valid query names eg. 'q9', 'q65', etc."""
    if not name.startswith("q"):
        return False
    try:
        q_num = int(name[1:])
    except ValueError:
        return False
    else:
        return 1 <= q_num <= 99


class PDSDSQueriesMeta(type):
    """Metaclass used for query lookup."""

    def __getattr__(cls, name: str):  # type: ignore[no-untyped-def]
        """Query lookup."""
        if valid_query(name):
            q_num = int(name[1:])
            module: ModuleType = importlib.import_module(
                f"cudf_polars.experimental.benchmarks.pdsds_queries.q{q_num}"
            )
            return getattr(module, cls.q_impl)
        raise AttributeError(f"{name} is not a valid query name")


class PDSDSQueries(metaclass=PDSDSQueriesMeta):
    """Base class for query loading."""

    q_impl: str
    name: str = "pdsds"


class PDSDSPolarsQueries(PDSDSQueries):
    """Polars Queries."""

    q_impl = "polars_impl"


class PDSDSDuckDBQueries(PDSDSQueries):
    """DuckDB Queries."""

    q_impl = "duckdb_impl"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PDS-DS benchmarks.")
    parser.add_argument(
        "--engine",
        choices=["polars", "duckdb", "validate"],
        default="polars",
        help="Which engine to use for executing the benchmarks or to validate results.",
    )
    args, extra_args = parser.parse_known_args()

    if args.engine == "polars":
        run_polars(PDSDSPolarsQueries, extra_args, num_queries=99)
    elif args.engine == "duckdb":
        run_duckdb(PDSDSDuckDBQueries, extra_args, num_queries=99)
    elif args.engine == "validate":
        run_validate(
            PDSDSPolarsQueries,
            PDSDSDuckDBQueries,
            extra_args,
            num_queries=99,
            check_dtypes=True,
            check_column_order=True,
        )
