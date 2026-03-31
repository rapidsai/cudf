# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
def test_fixed_rolling_basic(df, engine: pl.GPUEngine, rolling_fn):
    q = df.select(getattr(pl.col("x"), rolling_fn)(window_size=3))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("window_size", [1, 2, 4, 8])
def test_fixed_rolling_sum_window_sizes(df, engine: pl.GPUEngine, window_size):
    q = df.select(pl.col("x").rolling_sum(window_size=window_size))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_sum_centered(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_sum(window_size=3, center=True))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_sum_centered_even(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_sum(window_size=4, center=True))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_sum_min_samples(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_sum(window_size=3, min_samples=1))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_sum_with_nulls(engine: pl.GPUEngine):
    df = pl.LazyFrame({"x": [1.0, None, 3.0, None, 5.0, 6.0]})
    q = df.select(pl.col("x").rolling_sum(window_size=3, min_samples=1))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_sum_all_null_window(engine: pl.GPUEngine):
    df = pl.LazyFrame({"x": [None, None, None, 4.0, 5.0, 6.0]})
    q = df.select(pl.col("x").rolling_sum(window_size=3))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_mean_with_nulls(engine: pl.GPUEngine):
    df = pl.LazyFrame({"x": [1.0, None, 3.0, 4.0, None, 6.0]})
    q = df.select(pl.col("x").rolling_mean(window_size=3, min_samples=1))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_var(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_var(window_size=3))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_std(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_std(window_size=3))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_var_ddof(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_var(window_size=4, ddof=2))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_std_ddof(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_std(window_size=4, ddof=0))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.Float32, pl.Float64])
def test_fixed_rolling_sum_dtypes(engine: pl.GPUEngine, dtype):
    df = pl.LazyFrame({"x": pl.Series([1, 2, 3, 4, 5, 6], dtype=dtype)})
    q = df.select(pl.col("x").rolling_sum(window_size=3))
    assert_gpu_result_equal(q, engine=engine)


# TODO: Remove warning filter when fixed-size rolling supports streaming
@pytest.mark.filterwarnings(
    "ignore:This selection is not supported for multiple partitions\\.:UserWarning"
)
def test_fixed_rolling_large_window(engine: pl.GPUEngine):
    data = list(range(500))
    df = pl.LazyFrame({"x": [float(v) for v in data]})
    q = df.select(pl.col("x").rolling_sum(window_size=250))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_weighted_raises(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_mean(window_size=3, weights=[1.0, 2.0, 3.0]))
    assert_ir_translation_raises(q, engine, NotImplementedError)
