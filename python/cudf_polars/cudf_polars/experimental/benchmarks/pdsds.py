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
import dataclasses
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

with contextlib.suppress(ImportError):
    from cudf_polars.experimental.benchmarks.utils import (
        Record,
        RunConfig,
        execute_query,
        get_data,
        get_executor_options,
        initialize_dask_cluster,
        parse_args,
        print_query_plan,
    )

if TYPE_CHECKING:
    from collections.abc import Sequence

# Without this setting, the first IO task to run
# on each worker takes ~15 sec extra
os.environ["KVIKIO_COMPAT_MODE"] = os.environ.get("KVIKIO_COMPAT_MODE", "on")
os.environ["KVIKIO_NTHREADS"] = os.environ.get("KVIKIO_NTHREADS", "8")


class PDSDSDuckDBQueries:
    """PDS-DS query definitions."""

    @staticmethod
    def q1() -> str:
        """Query 1."""
        return """
        WITH customer_total_return
            AS (SELECT sr_customer_sk     AS ctr_customer_sk,
                        sr_store_sk        AS ctr_store_sk,
                        Sum(sr_return_amt) AS ctr_total_return
                FROM   store_returns,
                        date_dim
                WHERE  sr_returned_date_sk = d_date_sk
                        AND d_year = 2001
                GROUP  BY sr_customer_sk,
                        sr_store_sk)
        SELECT c_customer_id
        FROM   customer_total_return ctr1,
            store,
            customer
        WHERE  ctr1.ctr_total_return > (SELECT Avg(ctr_total_return) * 1.2
                                        FROM   customer_total_return ctr2
                                        WHERE  ctr1.ctr_store_sk = ctr2.ctr_store_sk)
            AND s_store_sk = ctr1.ctr_store_sk
            AND s_state = 'TN'
            AND ctr1.ctr_customer_sk = c_customer_sk
        ORDER  BY c_customer_id
        LIMIT 100;
        """


class PDSDSPolarsQueries:
    """PDS-DS query definitions."""

    @staticmethod
    def q0(run_config: RunConfig) -> pl.LazyFrame:
        """Query 0."""
        return pl.LazyFrame()

    @staticmethod
    def q1(run_config: RunConfig) -> pl.LazyFrame:
        """Query 1."""
        store_returns = get_data(
            run_config.dataset_path, "store_returns", run_config.suffix
        )
        date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
        store = get_data(run_config.dataset_path, "store", run_config.suffix)
        customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

        # Step 1: Create customer_total_return CTE equivalent
        customer_total_return = (
            store_returns.join(
                date_dim, left_on="sr_returned_date_sk", right_on="d_date_sk"
            )
            .filter(pl.col("d_year") == 2001)
            .group_by(["sr_customer_sk", "sr_store_sk"])
            .agg(
                [
                    pl.col("sr_customer_sk").first().alias("ctr_customer_sk"),
                    pl.col("sr_store_sk").first().alias("ctr_store_sk"),
                    pl.col("sr_return_amt").sum().alias("ctr_total_return"),
                ]
            )
        )

        # Step 2: Calculate average return per store for the subquery
        store_avg_returns = customer_total_return.group_by("ctr_store_sk").agg(
            [(pl.col("ctr_total_return").mean() * 1.2).alias("avg_return_threshold")]
        )

        # Step 3: Join everything together and apply filters
        return (
            customer_total_return.join(
                store_avg_returns, left_on="ctr_store_sk", right_on="ctr_store_sk"
            )
            .filter(pl.col("ctr_total_return") > pl.col("avg_return_threshold"))
            .join(store, left_on="ctr_store_sk", right_on="s_store_sk")
            .filter(pl.col("s_state") == "TN")
            .join(customer, left_on="ctr_customer_sk", right_on="c_customer_sk")
            .select(["c_customer_id"])
            .sort("c_customer_id")
            .limit(100)
        )


def execute_duckdb_query(query: str, dataset_path: Path) -> pl.DataFrame:
    """Execute a query with DuckDB."""
    import duckdb

    conn = duckdb.connect()

    create_statements = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".parquet"):
            table_name = filename.replace(".parquet", "")
            parquet_path = Path(dataset_path) / filename
            create_view_sql = f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{parquet_path}');"
            create_statements.append(create_view_sql)

    full_sql = "\n".join(create_statements) + "\n\n" + query
    return conn.execute(full_sql).pl()


def run_duckdb(options: Sequence[str] | None = None) -> None:
    """Run the benchmark with DuckDB."""
    args = parse_args(options, num_queries=99)
    run_config = RunConfig.from_args(args)

    for q_id in run_config.queries:
        try:
            duckdb_query = getattr(PDSDSDuckDBQueries, f"q{q_id}")()
        except AttributeError as err:
            raise NotImplementedError(f"Query {q_id} not implemented.") from err

        print(f"DuckDB Executing: {q_id}")
        t0 = time.time()
        result = execute_duckdb_query(duckdb_query, run_config.dataset_path)
        t1 = time.time()
        print(f"Completed {q_id} in {t1 - t0:.2f} seconds")

        if args.print_results:
            print(result)


def run_polars(options: Sequence[str] | None = None) -> None:
    """Run the benchmark with Polars."""
    args = parse_args(options, num_queries=99)
    run_config = RunConfig.from_args(args)
    client = initialize_dask_cluster(run_config, args)  # type: ignore

    records: defaultdict[int, list[Record]] = defaultdict(list)
    engine: pl.GPUEngine | None = None

    if run_config.executor != "cpu":
        executor_options: dict[str, Any] = {}
        if run_config.executor == "streaming":
            executor_options = get_executor_options(run_config)
        engine = pl.GPUEngine(
            raise_on_fail=True,
            executor=run_config.executor,
            executor_options=executor_options,
        )

    for q_id in run_config.queries:
        try:
            q = getattr(PDSDSPolarsQueries, f"q{q_id}")(run_config)
        except AttributeError as err:
            raise NotImplementedError(f"Query {q_id} not implemented.") from err

        print_query_plan(q_id, q, args, run_config, engine)

        records[q_id] = []

        for i in range(args.iterations):
            t0 = time.monotonic()

            result = execute_query(q_id, i, q, run_config, args, engine)

            t1 = time.monotonic()
            record = Record(query=q_id, duration=t1 - t0)
            if args.print_results:
                print(result)

            print(f"Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s")
            records[q_id].append(record)

    run_config = dataclasses.replace(run_config, records=dict(records))

    if args.summarize:
        run_config.summarize()

    if client is not None:
        client.close(timeout=60)

    args.output.write(json.dumps(run_config.serialize()))
    args.output.write("\n")


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
            executor_options=get_executor_options(run_config),
        )

    for q_id in run_config.queries:
        print(f"\nValidating Query {q_id}")
        try:
            polars_query = getattr(PDSDSPolarsQueries, f"q{q_id}")(run_config)
            duckdb_query = getattr(PDSDSDuckDBQueries, f"q{q_id}")()
        except AttributeError as err:
            raise NotImplementedError(f"Query {q_id} not implemented.") from err

        if baseline == "duckdb":
            base_result = execute_duckdb_query(duckdb_query, run_config.dataset_path)
        elif baseline == "cpu":
            base_result = polars_query.collect(new_streaming=True)

        if run_config.executor == "cpu":
            test_result = polars_query.collect(new_streaming=True)
        else:
            test_result = polars_query.collect(engine=engine)

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
        run_polars(extra_args)
    elif args.engine == "duckdb":
        run_duckdb(extra_args)
    elif args.engine == "validate":
        run_validate(extra_args)
