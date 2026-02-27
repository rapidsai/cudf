# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic GroupBy operations using the rapidsmpf runtime."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(scope="module")
def df():
    """Create a test DataFrame for groupby (keys + value columns)."""
    return pl.LazyFrame(
        {
            "key": list(range(50)) * 4,  # 200 rows, 50 unique keys
            "key2": list(range(10)) * 20,  # 200 rows, 10 unique
            "value": range(200),
            "value2": [1.0, 2.0, 3.0, 4.0, 5.0] * 40,
        }
    )


def _make_engine(target_partition_size: int = 1_000_000) -> pl.GPUEngine:
    """Create a GPUEngine with dynamic planning enabled."""
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 50,
            "runtime": "rapidsmpf",
            "dynamic_planning": {},
            "target_partition_size": target_partition_size,
        },
    )


@pytest.mark.parametrize("keys", [("key",), ("key", "key2")])
@pytest.mark.parametrize("agg", ["sum", "mean", "len", "min", "max"])
def test_dynamic_groupby_basic(df, keys, agg):
    """Test dynamic groupby with various key and agg combinations."""
    engine = _make_engine()
    expr = getattr(pl.col("value"), agg)()
    q = df.group_by(*keys).agg(expr)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_dynamic_groupby_tree_strategy(df):
    """Test that small output uses tree reduction (high target_partition_size)."""
    engine = _make_engine(target_partition_size=100_000_000)
    # Few keys -> small output -> tree reduction
    q = df.group_by("key2").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_dynamic_groupby_shuffle_strategy():
    """Test that large output uses shuffle (low target_partition_size)."""
    # High cardinality keys -> large output -> shuffle
    df = pl.LazyFrame({"key": range(1000), "value": range(1000)})
    engine = _make_engine(target_partition_size=1000)
    q = df.group_by("key").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_dynamic_groupby_single_group():
    """Test dynamic groupby where all rows have the same key."""
    engine = _make_engine()
    df = pl.LazyFrame({"key": [1] * 100, "value": range(100)})
    q = df.group_by("key").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_dynamic_groupby_multiple_aggs(df):
    """Test dynamic groupby with multiple aggregations."""
    engine = _make_engine()
    q = df.group_by("key").agg(
        pl.col("value").sum().alias("value_sum"),
        pl.col("value").mean().alias("value_mean"),
        pl.col("value2").min().alias("value2_min"),
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_dynamic_groupby_maintain_order(df):
    """Test dynamic groupby with maintain_order=True."""
    engine = _make_engine()
    q = df.group_by("key", maintain_order=True).agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_dynamic_groupby_single_row():
    """Test dynamic groupby on single-row DataFrame."""
    engine = _make_engine()
    df = pl.LazyFrame({"key": [1], "value": [42]})
    q = df.group_by("key").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
