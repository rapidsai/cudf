# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.experimental.explain import explain_query
from cudf_polars.testing.asserts import DEFAULT_SCHEDULER
from cudf_polars.testing.io import make_partitioned_source


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "x": range(25_000),
            "y": ["cat", "dog"] * 12_500,
            "z": [1.0, 2.0] * 12_500,
        }
    )


def test_explain_logical_plan(tmp_path, df):
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

    engine = pl.GPUEngine(executor="streaming", raise_on_fail=True)
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
            "scheduler": DEFAULT_SCHEDULER,
        },
    )

    plan = explain_query(q, engine)

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
            "scheduler": DEFAULT_SCHEDULER,
        },
    )

    plan = explain_query(q, engine, physical=True)

    assert "GROUPBY ('g',)" in plan


def test_explain_logical_plan_with_join(tmp_path, df):
    make_partitioned_source(df, tmp_path, fmt="parquet", n_files=2)

    left = pl.scan_parquet(tmp_path)
    right = pl.scan_parquet(tmp_path).select(["x", "z"]).rename({"z": "z2"})

    q = left.join(right, on="x", how="inner").select(["y", "z2"])

    engine = pl.GPUEngine(executor="streaming", raise_on_fail=True)
    plan = explain_query(q, engine, physical=False)

    assert "JOIN Inner ('x',) ('x',)" in plan


def test_explain_logical_plan_with_sort(tmp_path, df):
    make_partitioned_source(df, tmp_path, fmt="parquet", n_files=2)

    q = pl.scan_parquet(tmp_path).sort("z").select(["x", "z"])

    engine = pl.GPUEngine(executor="streaming", raise_on_fail=True)
    plan = explain_query(q, engine, physical=False)

    assert "SORT ('z',)" in plan


def test_explain_physical_plan_with_union_without_scan(df):
    q1 = df.lazy().select(["x", "z"])
    q2 = df.lazy().select(["x", "z"])
    q = pl.concat([q1, q2])

    engine = pl.GPUEngine(executor="streaming", raise_on_fail=True)
    plan = explain_query(q, engine, physical=False)

    assert "UNION" in plan


def test_explain_logical_plan_wide_table_with_scan(tmp_path):
    df = pl.DataFrame({f"col{i}": range(10) for i in range(10)})
    make_partitioned_source(df, tmp_path, fmt="parquet", n_files=1)

    q = pl.scan_parquet(tmp_path).select(df.columns)

    engine = pl.GPUEngine(executor="streaming", raise_on_fail=True)
    plan = explain_query(q, engine, physical=False)

    assert "SCAN PARQUET ('col0', 'col1', 'col2', '...', 'col8', 'col9')" in plan


def test_explain_logical_plan_wide_table():
    df = pl.DataFrame({f"col{i}": range(10) for i in range(20)})
    q = df.lazy().select(df.columns)

    engine = pl.GPUEngine(executor="streaming", raise_on_fail=True)
    plan = explain_query(q, engine, physical=False)

    assert "DATAFRAMESCAN ('col0', 'col1', 'col2', '...', 'col18', 'col19')" in plan
