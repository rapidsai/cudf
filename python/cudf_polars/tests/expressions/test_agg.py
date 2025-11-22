# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.dsl import expr
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture(
    params=[
        # regular aggs from Agg
        "min",
        "max",
        "median",
        "n_unique",
        "first",
        "last",
        "mean",
        "sum",
        "count",
        "std",
        "var",
        # scan aggs from UnaryFunction
        "cum_min",
        "cum_max",
        "cum_prod",
        "cum_sum",
    ]
)
def agg(request):
    return request.param


@pytest.fixture(params=[pl.Int32, pl.Float32, pl.Int16, pl.Int8, pl.UInt16])
def dtype(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["unsorted", "sorted"])
def is_sorted(request):
    return request.param


@pytest.fixture
def df(dtype, with_nulls, is_sorted):
    values = [-10, 4, 5, 2, 3, 6, 8, 9, 4, 4, 5, 2, 3, 7, 3, 6, -10, -11]
    if with_nulls:
        values = [None if v % 5 == 0 else v for v in values]

    if is_sorted:
        values = sorted(values, key=lambda x: -1000 if x is None else x)

    if dtype.is_unsigned_integer():
        values = pl.Series(values).abs()
        if is_sorted:
            values = values.sort()

    df = pl.LazyFrame({"a": values}, schema={"a": dtype})
    if is_sorted:
        return df.set_sorted("a")
    return df


@pytest.fixture
def decimal_df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": pl.Series(
                "a",
                [Decimal("0.10"), Decimal("1.10"), Decimal("100.10")],
                dtype=pl.Decimal(precision=9, scale=2),
            ),
        }
    )


def test_agg(df, agg):
    expr = getattr(pl.col("a"), agg)()
    q = df.select(expr)
    assert_gpu_result_equal(q, check_exact=False)


def test_bool_agg(agg, request):
    if agg == "cum_min" or agg == "cum_max":
        pytest.skip("Does not apply")
    df = pl.LazyFrame({"a": [True, False, None, True]})
    expr = getattr(pl.col("a"), agg)()
    q = df.select(expr)

    assert_gpu_result_equal(q, check_exact=False)


@pytest.mark.parametrize("cum_agg", sorted(expr.UnaryFunction._supported_cum_aggs))
def test_cum_agg_reverse_unsupported(cum_agg):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    expr = getattr(pl.col("a"), cum_agg)(reverse=True)
    q = df.select(expr)

    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize("q", [0.5, pl.lit(0.5)])
@pytest.mark.parametrize("interp", ["nearest", "higher", "lower", "midpoint", "linear"])
def test_quantile(df, q, interp):
    expr = pl.col("a").quantile(q, interp)
    q = df.select(expr)
    assert_gpu_result_equal(q, check_exact=False)


def test_quantile_invalid_q(df):
    expr = pl.col("a").quantile(pl.col("a"))
    q = df.select(expr)
    assert_ir_translation_raises(q, NotImplementedError)


def test_quantile_equiprobable_unsupported(df):
    expr = pl.col("a").quantile(0.5, interpolation="equiprobable")
    q = df.select(expr)
    assert_ir_translation_raises(q, NotImplementedError)


def test_quantile_duration_unsupported():
    df = pl.LazyFrame({"a": pl.Series([1, 2, 3, 4], dtype=pl.Duration("ns"))})
    q = df.select(pl.col("a").quantile(0.5))
    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize(
    "op", [pl.Expr.min, pl.Expr.nan_min, pl.Expr.max, pl.Expr.nan_max]
)
def test_agg_float_with_nans(op):
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, float("nan")], dtype=pl.Float64()),
            "b": pl.Series([1, 2, None], dtype=pl.Int8()),
        }
    )
    q = df.select(op(pl.col("a")), op(pl.col("b")))

    assert_gpu_result_equal(q)


@pytest.mark.xfail(reason="https://github.com/pola-rs/polars/issues/17513")
@pytest.mark.parametrize("op", [pl.Expr.max, pl.Expr.min])
def test_agg_singleton(op):
    df = pl.LazyFrame({"a": pl.Series([float("nan")])})

    q = df.select(op(pl.col("a")))

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("data", [[], [None], [None, 2, 3, None]])
def test_sum_empty_zero(data):
    df = pl.LazyFrame({"a": pl.Series(values=data, dtype=pl.Int32())})
    q = df.select(pl.col("a").sum())
    assert_gpu_result_equal(q)


def test_implode_agg_unsupported():
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, 3], dtype=pl.Int64()),
            "b": pl.Series([3, 4, 2], dtype=pl.Int64()),
            "c": pl.Series([1, None, 3], dtype=pl.Int64()),
            "d": pl.Series([10, None, 11], dtype=pl.Int64()),
        }
    )
    q = df.select(pl.col("b").implode())
    assert_ir_translation_raises(q, NotImplementedError)


def test_decimal_aggs(decimal_df: pl.LazyFrame) -> None:
    q = decimal_df.with_columns(
        sum=pl.col("a").sum(),
        min=pl.col("a").min(),
        max=pl.col("a").max(),
        mean=pl.col("a").mean(),
        median=pl.col("a").median(),
        mean_f32=pl.col("a").mean().cast(pl.Float32),
        median_f32=pl.col("a").median().cast(pl.Float32),
    )
    assert_gpu_result_equal(q)


def test_invalid_agg():
    df = pl.LazyFrame({"s": pl.Series(["a", "b", "c"], dtype=pl.String())})
    q = df.select(pl.col("s").sum())
    assert_ir_translation_raises(q, NotImplementedError)


def test_sum_all_null_decimal_dtype():
    df = pl.LazyFrame({"foo": pl.Series([None], dtype=pl.Decimal(9, 2))})
    q = df.select(pl.col("foo").sum())
    assert_gpu_result_equal(q)
