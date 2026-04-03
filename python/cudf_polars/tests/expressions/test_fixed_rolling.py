# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars import polars as plrs  # type: ignore[attr-defined]

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)

pytestmark = pytest.mark.skipif(
    not hasattr(plrs._expr_nodes, "RollingFunction"),
    reason="RollingFunction not available in this polars version",
)


@pytest.fixture
def df():
    return pl.LazyFrame(
        {
            "x": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        }
    )


@pytest.mark.parametrize(
    "rolling_fn",
    ["rolling_sum", "rolling_min", "rolling_max", "rolling_mean"],
)
def test_fixed_rolling_basic(df, rolling_fn):
    q = df.select(getattr(pl.col("x"), rolling_fn)(window_size=3))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("window_size", [1, 2, 4, 8])
def test_fixed_rolling_sum_window_sizes(df, window_size):
    q = df.select(pl.col("x").rolling_sum(window_size=window_size))
    assert_gpu_result_equal(q)


def test_fixed_rolling_sum_centered(df):
    q = df.select(pl.col("x").rolling_sum(window_size=3, center=True))
    assert_gpu_result_equal(q)


def test_fixed_rolling_sum_centered_even(df):
    q = df.select(pl.col("x").rolling_sum(window_size=4, center=True))
    assert_gpu_result_equal(q)


def test_fixed_rolling_sum_min_periods(df):
    q = df.select(pl.col("x").rolling_sum(window_size=3, min_periods=1))
    assert_gpu_result_equal(q)


def test_fixed_rolling_sum_with_nulls():
    df = pl.LazyFrame({"x": [1.0, None, 3.0, None, 5.0, 6.0]})
    q = df.select(pl.col("x").rolling_sum(window_size=3, min_periods=1))
    assert_gpu_result_equal(q)


def test_fixed_rolling_sum_all_null_window():
    df = pl.LazyFrame({"x": [None, None, None, 4.0, 5.0, 6.0]})
    q = df.select(pl.col("x").rolling_sum(window_size=3))
    assert_gpu_result_equal(q)


def test_fixed_rolling_mean_with_nulls():
    df = pl.LazyFrame({"x": [1.0, None, 3.0, 4.0, None, 6.0]})
    q = df.select(pl.col("x").rolling_mean(window_size=3, min_periods=1))
    assert_gpu_result_equal(q)


def test_fixed_rolling_var(df):
    q = df.select(pl.col("x").rolling_var(window_size=3))
    assert_gpu_result_equal(q)


def test_fixed_rolling_std(df):
    q = df.select(pl.col("x").rolling_std(window_size=3))
    assert_gpu_result_equal(q)


def test_fixed_rolling_var_ddof(df):
    q = df.select(pl.col("x").rolling_var(window_size=4, ddof=2))
    assert_gpu_result_equal(q)


def test_fixed_rolling_std_ddof(df):
    q = df.select(pl.col("x").rolling_std(window_size=4, ddof=0))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.Float32, pl.Float64])
def test_fixed_rolling_sum_dtypes(dtype):
    df = pl.LazyFrame({"x": pl.Series([1, 2, 3, 4, 5, 6], dtype=dtype)})
    q = df.select(pl.col("x").rolling_sum(window_size=3))
    assert_gpu_result_equal(q)


def test_fixed_rolling_large_window():
    data = list(range(500))
    df = pl.LazyFrame({"x": [float(v) for v in data]})
    q = df.select(pl.col("x").rolling_sum(window_size=250))
    assert_gpu_result_equal(q)


def test_fixed_rolling_weighted_raises(df):
    q = df.select(pl.col("x").rolling_mean(window_size=3, weights=[1.0, 2.0, 3.0]))
    assert_ir_translation_raises(q, NotImplementedError)
