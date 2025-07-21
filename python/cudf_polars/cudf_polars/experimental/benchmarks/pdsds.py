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
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

with contextlib.suppress(ImportError):
    from cudf_polars.experimental.benchmarks.utils import (
        Record,
        RunConfig,
        get_executor_options,
        parse_args,
        run_polars,
    )

if TYPE_CHECKING:
    from collections.abc import Sequence
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

    def __getattr__(cls, name: str):  # type: ignore
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


class PDSDSPolarsQueries(PDSDSQueries):
    """Polars Queries."""

    q_impl = "polars_impl"


class PDSDSDuckDBQueries(PDSDSQueries):
    """DuckDB Queries."""

    q_impl = "duckdb_impl"


def execute_duckdb_query(query: str, dataset_path: Path) -> pl.DataFrame:
    """Execute a query with DuckDB."""
    import duckdb

    conn = duckdb.connect()

    statements = [
        f"CREATE VIEW {table.stem} as SELECT * FROM read_parquet('{table.absolute()}');"
        for table in Path(dataset_path).glob("*.parquet")
    ]
    statements.append(query)
    return conn.execute("\n".join(statements)).pl()


def run_duckdb(options: Sequence[str] | None = None) -> None:
    """Run the benchmark with DuckDB."""
    args = parse_args(options, num_queries=99)
    run_config = RunConfig.from_args(args)
    records: defaultdict[int, list[Record]] = defaultdict(list)

    for q_id in run_config.queries:
        try:
            duckdb_query = getattr(PDSDSDuckDBQueries, f"q{q_id}")(run_config)
        except AttributeError as err:
            raise NotImplementedError(f"Query {q_id} not implemented.") from err

        print(f"DuckDB Executing: {q_id}")
        records[q_id] = []

        for i in range(args.iterations):
            t0 = time.time()

            result = execute_duckdb_query(duckdb_query, run_config.dataset_path)

            t1 = time.time()
            record = Record(query=q_id, duration=t1 - t0)
            if args.print_results:
                print(result)

            print(f"Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s")
            records[q_id].append(record)


def run_validate(options: Sequence[str] | None = None) -> None:
    """Validate Polars CPU vs DuckDB or Polars GPU."""
    from polars.testing import assert_frame_equal

    args = parse_args(options, num_queries=99)
    run_config = RunConfig.from_args(args)

    baseline = args.baseline
    if baseline not in {"duckdb", "cpu"}:
        raise ValueError("Baseline must be one of: 'duckdb', 'cpu'")

    failures: list[int] = []

    engine: pl.GPUEngine | None = None
    if run_config.executor != "cpu":
        engine = pl.GPUEngine(
            raise_on_fail=True,
            executor=run_config.executor,
            executor_options=get_executor_options(run_config, PDSDSPolarsQueries),
        )

    for q_id in run_config.queries:
        print(f"\nValidating Query {q_id}")
        try:
            polars_query = getattr(PDSDSPolarsQueries, f"q{q_id}")(run_config)
            duckdb_query = getattr(PDSDSDuckDBQueries, f"q{q_id}")(run_config)
        except AttributeError as err:
            raise NotImplementedError(f"Query {q_id} not implemented.") from err

        if baseline == "duckdb":
            base_result = execute_duckdb_query(duckdb_query, run_config.dataset_path)
        elif baseline == "cpu":
            base_result = polars_query.collect(new_streaming=True)

        if run_config.executor == "cpu":
            test_result = polars_query.collect(new_streaming=True)
        else:
            try:
                test_result = polars_query.collect(engine=engine)
            except Exception as e:
                failures.append(q_id)
                print(f"❌ Query {q_id} failed validation: GPU execution failed.\n{e}")
                continue

        try:
            assert_frame_equal(
                base_result,
                test_result,
                check_dtypes=True,
                check_column_order=False,
            )
            print(f"✅ Query {q_id} passed validation.")
        except AssertionError as e:
            failures.append(q_id)
            print(f"❌ Query {q_id} failed validation:\n{e}")
            if args.print_results:
                print("Baseline Result:\n", base_result)
                print("Test Result:\n", test_result)

    if failures:
        print("\nValidation Summary:")
        print("===================")
        print(f"{len(failures)} query(s) failed: {failures}")
    else:
        print("\nAll queries passed validation.")


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
        run_duckdb(extra_args)
    elif args.engine == "validate":
        run_validate(extra_args)
