# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CSE HStack handling in the streaming executor."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame({"a": [1, 2, 3, 4, 5, 6, 7], "b": [2, 3, 4, 5, 6, 7, 8]})


def test_cse_agg_with_columns(df, engine):
    # CSE: col("a").sum() appears twice; Polars emits HStack(broadcast=False)
    # with the shared aggregation, then HStack(broadcast=True) with the outer
    # expressions. Exercises the Projection handler's outer_bcast_hstacks path.
    q = df.with_columns(
        pl.col("a").sum().alias("s"),
        (pl.col("a").sum() * 2).alias("s2"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_cse_agg_select(df, engine):
    # CSE: col("a").sum() appears twice in a select.
    # Exercises the Select handler with no outer broadcast HStack.
    q = df.select(
        pl.col("a").sum().alias("s"),
        (pl.col("a").sum() * 2).alias("s2"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_hstack_non_scalar_cse_fallback(df, engine):
    # Non-scalar CSE (head(5)) skips the CSE transform, falling through to the
    # non-pointwise HStack fallback in lower_ir_node.register(HStack).
    q = df.with_columns(
        pl.col("a").head(5).min().alias("min_5"),
        pl.col("a").head(5).max().alias("max_5"),
    )
    with pytest.warns(UserWarning, match="not supported for multiple partitions"):
        assert_gpu_result_equal(q, engine=engine)
