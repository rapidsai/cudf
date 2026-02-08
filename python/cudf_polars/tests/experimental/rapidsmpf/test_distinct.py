# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic Distinct node in rapidsmpf runtime."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(scope="module")
def df():
    """Create a test DataFrame with duplicates."""
    return pl.LazyFrame(
        {
            "x": list(range(50)) * 4,  # 200 rows, 50 unique
            "y": list(range(10)) * 20,  # 200 rows, 10 unique
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 40,
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
            "dynamic_planning": {
                "sample_chunk_count": 2,
            },
            "target_partition_size": target_partition_size,
        },
    )


@pytest.mark.parametrize("subset", [None, ("y",), ("x", "y")])
@pytest.mark.parametrize("keep", ["any", "none"])
def test_dynamic_distinct_basic(df, subset, keep):
    """Test dynamic distinct with various subset and keep options."""
    engine = _make_engine()
    q = df.unique(subset=subset, keep=keep, maintain_order=False)

    # For keep="any" with subset, select only the subset columns
    # to avoid non-deterministic results for non-key columns
    if keep == "any" and subset:
        q = q.select(*(pl.col(col) for col in subset))

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_dynamic_distinct_tree_strategy(df):
    """Test that small output uses tree reduction (high target_partition_size)."""
    # With a very high target_partition_size, tree reduction should be chosen
    engine = _make_engine(target_partition_size=100_000_000)
    subset = ("y",)
    q = df.unique(subset=subset, keep="any", maintain_order=False)
    # With keep="any", non-subset columns are non-deterministic, so only check subset
    q = q.select(*(pl.col(col) for col in subset))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_dynamic_distinct_shuffle_strategy():
    """Test that large output uses shuffle (low target_partition_size)."""
    # Create a DataFrame with high cardinality
    df = pl.LazyFrame({"x": range(1000), "y": range(1000)})

    # With a very low target_partition_size, shuffle should be chosen
    engine = _make_engine(target_partition_size=1000)
    q = df.unique(subset=None, keep="any", maintain_order=False)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_dynamic_distinct_all_duplicates():
    """Test dynamic distinct where all rows are duplicates."""
    engine = _make_engine()
    # All rows are identical, so result should be a single row
    df = pl.LazyFrame({"x": [1, 1, 1, 1], "y": [2, 2, 2, 2]})
    q = df.unique()
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_dynamic_distinct_single_row():
    """Test dynamic distinct on single-row DataFrame."""
    engine = _make_engine()
    df = pl.LazyFrame({"x": [1], "y": [2]})
    q = df.unique()
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
