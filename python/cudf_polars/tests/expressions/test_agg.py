# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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


def test_agg(df, agg):
    expr = getattr(pl.col("a"), agg)()
    q = df.select(expr)

    # https://github.com/rapidsai/cudf/issues/15852
    check_dtypes = agg not in {"n_unique", "median"}
    if not check_dtypes and q.collect_schema()["a"] != pl.Float64:
        with pytest.raises(AssertionError):
            assert_gpu_result_equal(q)
    assert_gpu_result_equal(q, check_dtypes=check_dtypes, check_exact=False)


def test_bool_agg(agg, request):
    if agg == "cum_min" or agg == "cum_max":
        pytest.skip("Does not apply")
    request.applymarker(
        pytest.mark.xfail(
            condition=agg == "n_unique",
            reason="Wrong dtype we get Int32, polars gets UInt32",
        )
    )
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

    # https://github.com/rapidsai/cudf/issues/15852
    check_dtypes = q.collect_schema()["a"] == pl.Float64
    if not check_dtypes:
        with pytest.raises(AssertionError):
            assert_gpu_result_equal(q)
    assert_gpu_result_equal(q, check_dtypes=check_dtypes, check_exact=False)


def test_quantile_invalid_q(df):
    expr = pl.col("a").quantile(pl.col("a"))
    q = df.select(expr)
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
