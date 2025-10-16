# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest

import polars as pl

from cudf_polars.experimental.explain import _fmt_row_count, explain_query
from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)
from cudf_polars.testing.io import make_lazy_frame, make_partitioned_source


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "x": range(25_000),
            "y": ["cat", "dog"] * 12_500,
            "z": [1.0, 2.0] * 12_500,
        }
    )


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "shuffle_method": DEFAULT_RUNTIME,  # Names coincide
            "target_partition_size": 10_000,
            "max_rows_per_partition": 1_000,
            "stats_planning": {
                "use_reduction_planning": True,
                "default_selectivity": 0.5,
            },
        },
    )


@pytest.mark.parametrize("executor", ["streaming", "in-memory"])
def test_explain_logical_plan(tmp_path, df, executor):
    make_partitioned_source(df, tmp_path, fmt="parquet", n_files=2)

    q = (
        pl.scan_parquet(tmp_path)
        .filter((pl.col("x") > 10) & (pl.col("y") == "dog"))
        .with_columns(
            (pl.col("x") * pl.col("z")).alias("xz"),
            (pl.col("x") % 5).alias("bucket"),
        )
        .group_by("bucket")
        .agg(pl.sum("xz").alias("sum_xz"))
        .select(pl.col("sum_xz"))
    )

    engine = pl.GPUEngine(executor=executor, raise_on_fail=True)
    plan = explain_query(q, engine, physical=False)

    assert "SCAN PARQUET" in plan
    assert "SELECT" in plan
    assert "PROJECTION" in plan
    assert "GROUPBY ('bucket',)" in plan


def test_explain_physical_plan(tmp_path, df):
    make_partitioned_source(df, tmp_path, fmt="parquet", n_files=5)

    q = (
        pl.scan_parquet(tmp_path)
        .filter((pl.col("x") < 40_000) & (pl.col("z") > 1.0))
        .with_columns((pl.col("x") + pl.col("z")).alias("sum"))
        .select(["sum", "y"])
    )

    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={
            "target_partition_size": 10_000,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )

    plan = explain_query(q, engine)

    if DEFAULT_RUNTIME == "tasks":
        # rapidsmpf runtime does not split Scan nodes at lowering time
        assert "UNION" in plan
        assert "SPLITSCAN" in plan
    assert "SELECT ('sum', 'y')" in plan or "PROJECTION ('sum', 'y')" in plan


def test_explain_physical_plan_with_groupby(tmp_path, df):
    make_partitioned_source(df, tmp_path, fmt="parquet", n_files=1)

    q = (
        pl.scan_parquet(tmp_path)
        .with_columns((pl.col("x") % 3).alias("g"))
        .group_by("g")
        .agg(pl.len())
    )

    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={
            "target_partition_size": 10_000,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )

    plan = explain_query(q, engine, physical=True)

    assert "GROUPBY ('g',)" in plan


def test_explain_logical_plan_with_join(tmp_path, df):
    make_partitioned_source(df, tmp_path, fmt="parquet", n_files=2)

    left = pl.scan_parquet(tmp_path)
    right = pl.scan_parquet(tmp_path).select(["x", "z"]).rename({"z": "z2"})

    q = left.join(right, on="x", how="inner").select(["y", "z2"])

    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={"runtime": DEFAULT_RUNTIME},
    )
    plan = explain_query(q, engine, physical=False)

    assert "JOIN Inner ('x',) ('x',)" in plan


def test_explain_logical_plan_with_sort(tmp_path, df):
    make_partitioned_source(df, tmp_path, fmt="parquet", n_files=2)

    q = pl.scan_parquet(tmp_path).sort("z").select(["x", "z"])

    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={"runtime": DEFAULT_RUNTIME},
    )
    plan = explain_query(q, engine, physical=False)

    assert "SORT ('z',)" in plan


def test_explain_physical_plan_with_union_without_scan(df):
    q1 = df.lazy().select(["x", "z"])
    q2 = df.lazy().select(["x", "z"])
    q = pl.concat([q1, q2])

    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={"runtime": DEFAULT_RUNTIME},
    )
    plan = explain_query(q, engine, physical=False)

    assert "UNION" in plan


def test_explain_logical_plan_wide_table_with_scan(tmp_path):
    df = pl.DataFrame({f"col{i}": range(10) for i in range(10)})
    make_partitioned_source(df, tmp_path, fmt="parquet", n_files=1)

    q = pl.scan_parquet(tmp_path).select(df.columns)

    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={"runtime": DEFAULT_RUNTIME},
    )
    plan = explain_query(q, engine, physical=False)

    assert "SCAN PARQUET ('col0', 'col1', 'col2', '...', 'col8', 'col9')" in plan


def test_explain_logical_plan_wide_table():
    df = pl.DataFrame({f"col{i}": range(10) for i in range(20)})
    q = df.lazy().select(df.columns)

    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
        executor_options={"runtime": DEFAULT_RUNTIME},
    )
    plan = explain_query(q, engine, physical=False)

    assert "DATAFRAMESCAN ('col0', 'col1', 'col2', '...', 'col18', 'col19')" in plan


def test_fmt_row_count():
    assert _fmt_row_count(None) == ""
    assert _fmt_row_count(0) == "0"
    assert _fmt_row_count(1000) == "1 K"
    assert _fmt_row_count(1_234_000) == "1.23 M"
    assert _fmt_row_count(1_250_000_000) == "1.25 B"


@pytest.mark.parametrize("kind", ["parquet", "csv", "frame"])
@pytest.mark.parametrize("n_rows", [None, 3])
@pytest.mark.parametrize("select", [True, False])
def test_explain_logical_io_then_distinct(engine, tmp_path, kind, n_rows, select):
    # Create simple Distinct or Select(unique) + Sort query
    df0 = pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "customer_id": [101, 102, 101, 103, 103, 104],
            "amount": [50.0, 75.0, 30.0, 120.0, 85.0, 40.0],
            "year": [2023, 2023, 2023, 2024, 2024, 2024],
        }
    )
    df = make_lazy_frame(df0, kind, path=tmp_path, n_files=2, n_rows=n_rows)
    if select:
        q = df.select(pl.col("customer_id").unique()).sort("customer_id")
    else:
        q = df.unique(subset=["customer_id"]).sort("order_id")

    # Verify the query runs correctly
    # TODO: Is the cpu engine doing the right thing here?
    assert_gpu_result_equal(q, engine=engine)

    # Check query plan
    repr = explain_query(q, engine, physical=False)
    if kind == "csv":
        # CSV will NOT provide row-count statistics unless n_rows is provided,
        # and it will never provide unique-count statistics.
        if n_rows is None:
            assert re.search(r"^\s*SORT.*row_count='unknown'\s*$", repr, re.MULTILINE)
        else:
            value = _fmt_row_count(n_rows)
            assert re.search(
                rf"^\s*SORT.*row_count=\'~{value}\'\s*$", repr, re.MULTILINE
            )
    else:
        value = _fmt_row_count(q.collect().height)
        assert re.search(rf"^\s*SORT.*row_count=\'~{value}\'\s*$", repr, re.MULTILINE)


@pytest.mark.parametrize("kind", ["parquet", "csv", "frame"])
def test_explain_logical_io_then_filter(engine, tmp_path, kind):
    # Create simple Distinct or Select(unique) + Sort query.
    # NOTE: This test depends on a "default_selectivity" of 0.5
    # and a very-specific DataFrame and predicate.
    df = pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "customer_id": [101, 102, 101, 103, 104, 104, 105, 106],
        }
    )
    df = make_lazy_frame(df, kind, path=tmp_path, n_files=2)
    # Include `unique` to improve code coverage
    q = (
        df.filter(pl.col("customer_id") < 104)
        .unique(subset=["order_id"])
        .sort("order_id")
    )

    # Verify the query runs correctly
    assert_gpu_result_equal(q, engine=engine)

    # Check query plan
    repr = explain_query(q, engine, physical=False)
    if kind == "csv":
        assert re.search(r"^\s*SORT.*row_count='unknown'\s*$", repr, re.MULTILINE)
    else:
        value = _fmt_row_count(q.collect().height)
        assert re.search(rf"^\s*SORT.*row_count=\'~{value}\'\s*$", repr, re.MULTILINE)


@pytest.mark.parametrize("kind", ["parquet", "csv", "frame"])
def test_explain_logical_agg(engine, tmp_path, kind):
    # Create simple aggregation query
    df = pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "customer_id": [101, 102, 101, 103, 103, 104],
            "amount": [50.0, 75.0, 30.0, 120.0, 85.0, 40.0],
            "year": [2023, 2023, 2023, 2024, 2024, 2024],
        }
    )
    df = make_lazy_frame(df, kind, path=tmp_path, n_files=2)
    q = df.select(pl.sum("customer_id"))

    # Verify the query runs correctly
    assert_gpu_result_equal(q, engine=engine)

    # Check query plan - We should know that sum produces a single row.
    repr = explain_query(q, engine, physical=False)
    assert re.search(
        rf"^\s*SELECT.*row_count=\'~{q.collect().height}\'\s*$", repr, re.MULTILINE
    )


@pytest.mark.parametrize("kind", ["parquet", "csv", "frame"])
def test_explain_logical_io_then_join(engine, tmp_path, kind):
    # Create simple Join + Sort query
    sales = pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "customer_id": [101, 102, 101, 103, 102, 101],
            "amount": [50.0, 75.0, 30.0, 120.0, 85.0, 40.0],
            "product": ["A", "B", "A", "C", "B", "A"],
        }
    )
    sales = make_lazy_frame(sales, kind, path=tmp_path / f"sales_{kind}")
    customers = pl.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "customer_name": ["Alice", "Bob", "Charlie"],
            "region": ["North", "South", "North"],
        }
    )
    customers = make_lazy_frame(customers, kind, path=tmp_path / f"customers_{kind}")
    q = sales.join(customers, on="customer_id", how="inner").sort("customer_id")

    # Verify the query runs correctly
    assert_gpu_result_equal(q, engine=engine)

    # Check the query plan
    repr = explain_query(q, engine, physical=False)
    if kind == "csv":
        assert re.search(r"^\s*SORT.*row_count='unknown'\s*$", repr, re.MULTILINE)
    else:
        value = _fmt_row_count(q.collect().height)
        assert re.search(rf"^\s*SORT.*row_count=\'~{value}\'\s*$", repr, re.MULTILINE)


@pytest.mark.parametrize("kind", ["parquet", "csv", "frame"])
def test_explain_logical_io_then_join_then_groupby(engine, tmp_path, kind):
    # Create simple Join + GroupBy + Sort query
    sales = pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "customer_id": [101, 102, 101, 103, 102, 101],
            "amount": [50.0, 75.0, 30.0, 120.0, 85.0, 40.0],
            "product": ["A", "B", "A", "C", "B", "A"],
        }
    )
    sales = make_lazy_frame(sales, kind, path=tmp_path / f"sales_{kind}")
    customers = pl.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "customer_name": ["Alice", "Bob", "Charlie"],
            "region": ["North", "South", "North"],
        }
    )
    customers = make_lazy_frame(customers, kind, path=tmp_path / f"customers_{kind}")
    q_join = sales.join(customers, on="customer_id", how="inner")
    q_gb = q_join.group_by("customer_id").agg(
        [
            pl.col("amount").sum().alias("total_amount"),
            pl.col("order_id").count().alias("order_count"),
            pl.col("customer_name").first().alias("name"),
            pl.col("region").first().alias("region"),
        ]
    )
    q = q_gb.sort("customer_id")

    # Verify the query runs correctly
    assert_gpu_result_equal(q, engine=engine)

    # Check the query plan
    repr = explain_query(q, engine, physical=False)
    if kind == "csv":
        assert re.search(r"^\s*SORT.*row_count='unknown'\s*$", repr, re.MULTILINE)
    else:
        join_count = _fmt_row_count(q_join.collect().height)
        gb_count = _fmt_row_count(q_gb.collect().height)
        final_count = _fmt_row_count(q.collect().height)
        assert re.search(
            rf"^\s*GROUPBY.*row_count=\'~{gb_count}\'\s*$", repr, re.MULTILINE
        )
        assert re.search(
            rf"^\s*JOIN Inner.*row_count=\'~{join_count}\'\s*$", repr, re.MULTILINE
        )
        assert re.search(
            rf"^\s*SORT.*row_count=\'~{final_count}\'\s*$", repr, re.MULTILINE
        )


@pytest.mark.parametrize("kind", ["parquet", "csv", "frame"])
def test_explain_logical_io_then_concat_then_groupby(engine, tmp_path, kind):
    # Create first table - sales data from 2023
    sales_2023 = pl.DataFrame(
        {
            "order_id": [1, 2, 3],
            "customer_id": [101, 102, 101],
            "amount": [50.0, 75.0, 30.0],
            "year": [2023, 2023, 2023],
        }
    )
    df_2023 = make_lazy_frame(sales_2023, kind, path=tmp_path / f"sales_2023.{kind}")

    # Create second table - sales data from 2024
    sales_2024 = pl.DataFrame(
        {
            "order_id": [4, 5, 6],
            "customer_id": [103, 103, 104],
            "amount": [120.0, 85.0, 40.0],
            "year": [2024, 2024, 2024],
        }
    )
    df_2024 = make_lazy_frame(sales_2024, kind, path=tmp_path / f"sales_2024.{kind}")

    # Create second table - sales data from 2025
    sales_2025 = pl.DataFrame(
        {
            "order_id": [7, 8, 9],
            "customer_id": [105, 109, 106],
            "amount": [10.0, 50.0, 100.0],
            "year": [2025, 2025, 2025],
        }
    )
    df_2025 = make_lazy_frame(sales_2025, kind, path=tmp_path / f"sales_2025.{kind}")

    def _gb(df):
        return df.group_by("customer_id").agg(
            [
                pl.col("amount").sum().alias("total_amount"),
                pl.col("order_id").count().alias("order_count"),
                pl.col("year").min().alias("first_year"),
                pl.col("year").max().alias("last_year"),
            ]
        )

    # Group by customer_id after concatenation
    q_concat_1 = pl.concat([df_2023, df_2024])
    q_gb_1 = _gb(q_concat_1)
    q_1 = q_gb_1.sort("customer_id").head(2)
    q_gb_2 = _gb(df_2025)
    q_concat_2 = pl.concat([q_gb_1, q_gb_2])
    q_2 = q_concat_2.sort("customer_id").head(2)

    # Verify the query runs correctly
    assert_gpu_result_equal(q_1, engine=engine)
    assert_gpu_result_equal(q_2, engine=engine)

    # Check query plan q_1
    repr = explain_query(q_1, engine, physical=False)
    if kind == "csv":
        assert re.search(r"^\s*SORT.*row_count='unknown'\s*$", repr, re.MULTILINE)
    else:
        concat_count_1 = _fmt_row_count(q_concat_1.collect().height)
        gb_count_1 = _fmt_row_count(q_gb_1.collect().height)
        final_count_1 = _fmt_row_count(q_1.collect().height)
        assert re.search(
            rf"^\s*UNION.*row_count=\'~{concat_count_1}\'\s*$", repr, re.MULTILINE
        )
        assert re.search(
            rf"^\s*GROUPBY.*row_count=\'~{gb_count_1}\'\s*$", repr, re.MULTILINE
        )
        assert re.search(
            rf"^\s*SORT.*row_count=\'~{final_count_1}\'\s*$", repr, re.MULTILINE
        )

    # Check query plan q_2
    repr = explain_query(q_2, engine, physical=False)
    if kind == "csv":
        assert re.search(r"^\s*SORT.*row_count='unknown'\s*$", repr, re.MULTILINE)
    else:
        concat_count_1 = _fmt_row_count(q_concat_1.collect().height)
        concat_count_2 = _fmt_row_count(q_concat_2.collect().height)
        gb_count_1 = _fmt_row_count(q_gb_1.collect().height)
        gb_count_2 = _fmt_row_count(q_gb_2.collect().height)
        final_count_2 = _fmt_row_count(q_2.collect().height)
        assert re.search(
            rf"^\s*UNION.*row_count=\'~{concat_count_1}\'\s*$", repr, re.MULTILINE
        )
        assert re.search(
            rf"^\s*UNION.*row_count=\'~{concat_count_2}\'\s*$", repr, re.MULTILINE
        )
        assert re.search(
            rf"^\s*GROUPBY.*row_count=\'~{gb_count_1}\'\s*$", repr, re.MULTILINE
        )
        assert re.search(
            rf"^\s*GROUPBY.*row_count=\'~{gb_count_2}\'\s*$", repr, re.MULTILINE
        )
        assert re.search(
            rf"^\s*SORT.*row_count=\'~{final_count_2}\'\s*$", repr, re.MULTILINE
        )
