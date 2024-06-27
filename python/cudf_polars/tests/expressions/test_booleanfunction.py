# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(params=[False, True], ids=["no_nulls", "nulls"])
def has_nulls(request):
    return request.param


@pytest.mark.parametrize(
    "ignore_nulls",
    [
        pytest.param(
            False, marks=pytest.mark.xfail(reason="No support for Kleene logic")
        ),
        True,
    ],
)
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
def test_boolean_function_unary(request, expr, has_nans, has_nulls):
    if has_nulls and expr in (pl.Expr.is_nan, pl.Expr.is_not_nan):
        request.applymarker(
            pytest.mark.xfail(
                reason="Need to copy null mask since is_{not_}nan(null) => null"
            )
        )

    values: list[float | None] = [1, 2, 3, 4, 5]
    if has_nans:
        values[3] = float("nan")
    if has_nulls:
        values[0] = None

    df = pl.LazyFrame({"a": pl.Series(values, dtype=pl.Float32())})

    q = df.select(expr(pl.col("a")))

    assert_gpu_result_equal(q)


@pytest.mark.xfail(reason="Evaluation handlers not yet implemented")
@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").is_finite(),
        pl.col("a").is_infinite(),
        pl.col("a").is_in(pl.col("b")),
    ],
)
def test_unsupported_boolean_function(expr):
    df = pl.LazyFrame(
        {"a": pl.Series([1, float("nan"), 2, 4], dtype=pl.Float64()), "b": [1, 2, 3, 4]}
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
def test_boolean_horizontal(request, expr, has_nulls, wide):
    if has_nulls:
        request.applymarker(pytest.mark.xfail(reason="No support for Kleene logic"))
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
