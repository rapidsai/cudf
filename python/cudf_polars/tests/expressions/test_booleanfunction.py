# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture(params=[False, True], ids=["no_nulls", "nulls"])
def has_nulls(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["include_nulls", "ignore_nulls"])
def ignore_nulls(request):
    return request.param


def test_booleanfunction_reduction(ignore_nulls):
    ldf = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, 3.0, 2, 5], dtype=pl.Float64()),
            "b": [0, 3, 1, -1, None],
            "c": [1, 6, 5, 3, 2],
        }
    )

    query = ldf.select(
        (pl.col("a") > 3).any(ignore_nulls=ignore_nulls),
        (pl.col("b") > 2).all(ignore_nulls=ignore_nulls),
    )

    assert_gpu_result_equal(query)


@pytest.mark.parametrize("expr", [pl.Expr.any, pl.Expr.all])
def test_booleanfunction_all_any_kleene(expr, ignore_nulls):
    ldf = pl.LazyFrame(
        {
            "a": [False, None],
            "b": [False, False],
            "c": [False, True],
            "d": [None, False],
            "e": pl.Series([None, None], dtype=pl.Boolean()),
            "f": [None, True],
            "g": [True, False],
            "h": [True, None],
            "i": [True, True],
        }
    )
    q = ldf.select(expr(pl.col("*"), ignore_nulls=ignore_nulls))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "expr",
    [
        pl.Expr.is_null,
        pl.Expr.is_not_null,
        pl.Expr.is_nan,
        pl.Expr.is_not_nan,
    ],
    ids=lambda f: f"{f.__name__}()",
)
@pytest.mark.parametrize("has_nans", [False, True], ids=["no_nans", "nans"])
def test_boolean_function_unary(expr, has_nans, has_nulls):
    values: list[float | None] = [1, 2, 3, 4, 5]
    if has_nans:
        values[3] = float("nan")
    if has_nulls:
        values[0] = None

    df = pl.LazyFrame({"a": pl.Series(values, dtype=pl.Float32())})

    q = df.select(expr(pl.col("a")), expr(pl.col("a")).not_().alias("b"))

    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").is_finite(),
        pl.col("a").is_infinite(),
        [pl.col("a").is_infinite(), pl.col("b").is_finite()],
    ],
)
def test_boolean_finite(expr):
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, float("nan"), 2, float("inf")], dtype=pl.Float64()),
            "b": [1, 2, 3, 4],
            "c": pl.Series([1, 2, 3, 4], dtype=pl.Float64()),
        }
    )

    q = df.select(expr)

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("closed", ["both", "left", "right", "none"])
@pytest.mark.parametrize(
    "bounds", [(1, 2), (-1, 10), (11, 10), (pl.col("lo"), pl.col("hi"))]
)
def test_boolean_isbetween(closed, bounds):
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, float("nan"), 2, 4], dtype=pl.Float32()),
            "lo": [1, 2, 2, 3],
            "hi": [10, 4, 2, 4],
        }
    )

    q = df.select(pl.col("a").is_between(*bounds, closed=closed))

    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "expr", [pl.any_horizontal("*"), pl.all_horizontal("*")], ids=["any", "all"]
)
@pytest.mark.parametrize("wide", [False, True], ids=["narrow", "wide"])
def test_boolean_horizontal(expr, has_nulls, wide):
    ldf = pl.LazyFrame(
        {
            "a": [False, False, False, False, False, True],
            "b": [False, False, True, None, None, True],
            "c": [False, True, True, False, True, True],
        }
    )
    if not has_nulls:
        ldf = ldf.select(pl.col("a"), pl.col("c"))

    # To see the All/Any Horizontal nodes, we need a dataframe with
    # more than 128 columns
    if wide:
        ldf = ldf.with_columns(pl.col("c").alias(f"col{i}") for i in range(128))
    q = ldf.select(expr)

    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").is_in(pl.col("b")),
        pl.col("a").is_in(pl.col("c")),
        pl.col("c").is_in(pl.col("d")),
    ],
)
def test_boolean_is_in(expr):
    ldf = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, 3], dtype=pl.Int64()),
            "b": pl.Series([3, 4, 2], dtype=pl.Int64()),
            "c": pl.Series([1, None, 3], dtype=pl.Int64()),
            "d": pl.Series([10, None, 11], dtype=pl.Int64()),
        }
    )

    q = ldf.select(expr)

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("expr", [pl.Expr.and_, pl.Expr.or_, pl.Expr.xor])
def test_boolean_kleene_logic(expr):
    ldf = pl.LazyFrame(
        {
            "a": [False, False, False, None, None, None, True, True, True],
            "b": [False, None, True, False, None, True, False, None, True],
        }
    )
    q = ldf.select(expr(pl.col("a"), pl.col("b")))
    assert_gpu_result_equal(q)


def test_boolean_is_in_raises_unsupported():
    ldf = pl.LazyFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int64)})
    q = ldf.select(pl.col("a").is_in(pl.lit(1, dtype=pl.Int32())))

    assert_ir_translation_raises(q, NotImplementedError)
