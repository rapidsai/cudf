# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_132

if TYPE_CHECKING:
    from cudf_polars.typing import RankMethod, RoundMethod


@pytest.fixture(
    params=[
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "arcsinh",
        "arccosh",
        "arctanh",
        "exp",
        "sqrt",
        "cbrt",
        "ceil",
        "floor",
        "abs",
    ]
)
def op(request):
    return request.param


@pytest.fixture(params=[pl.Int32, pl.Float32])
def dtype(request):
    return request.param


@pytest.fixture
def ldf(with_nulls, dtype):
    values = [1, 2, 4, 5, -2, -4, 0]
    if with_nulls:
        values.append(None)
    if dtype == pl.Float32:
        values.append(-float("inf"))
        values.append(float("nan"))
        values.append(float("inf"))
    elif dtype == pl.Int32:
        values.append(-2_147_483_648)
        values.append(2_147_483_647)
    return pl.LazyFrame(
        {
            "a": pl.Series(values, dtype=dtype),
            "b": pl.Series([i - 4 for i in range(len(values))], dtype=pl.Float32),
            "c": pl.Series([timedelta(hours=i) for i in range(len(values))]),
        }
    )


def test_unary(ldf, op):
    expr = getattr(pl.col("a"), op)()
    q = ldf.select(expr)
    assert_gpu_result_equal(q, check_exact=False)


@pytest.mark.parametrize("base_literal", [False, True])
@pytest.mark.parametrize("exponent_literal", [False, True])
def test_pow(ldf, base_literal, exponent_literal):
    base = pl.lit(2) if base_literal else pl.col("a")
    exponent = pl.lit(-3, dtype=pl.Float32) if exponent_literal else pl.col("b")

    q = ldf.select(base.pow(exponent))

    assert_gpu_result_equal(q, check_exact=False)


@pytest.mark.parametrize("natural", [True, False])
def test_log(ldf, natural):
    if natural:
        expr = pl.col("a").log()
    else:
        expr = pl.col("a").log(10)

    q = ldf.select(expr)

    assert_gpu_result_equal(q, check_exact=False)


@pytest.mark.parametrize("col", ["a", "b", "c"])
def test_negate(ldf, col):
    q = ldf.select(-pl.col(col))
    assert_gpu_result_equal(q)


def test_null_count():
    lf = pl.LazyFrame(
        {
            "foo": [1, None, 3],
            "bar": [None, None, 1],
            "baz": [1, 2, 3],
        }
    )
    q = lf.select(
        pl.col("foo").is_null().sum(),
        pl.col("bar").is_null().sum(),
        pl.col("baz").is_null().sum(),
    )
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
def test_rank_supported(
    request, ldf: pl.LazyFrame, method: RankMethod, *, descending: bool
):
    request.applymarker(
        pytest.mark.xfail(condition=POLARS_VERSION_LT_132, reason="rank unsupported")
    )
    expr = pl.col("a").rank(method=method, descending=descending)
    q = ldf.select(expr)
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("test", ["with_nulls", "with_ties"])
def test_rank_methods_with_nulls_or_ties(
    request, ldf: pl.LazyFrame, method: RankMethod, *, descending: bool, test: str
) -> None:
    request.applymarker(
        pytest.mark.xfail(condition=POLARS_VERSION_LT_132, reason="rank unsupported")
    )

    base = pl.col("a")
    if test == "with_nulls":
        expr = pl.when((base % 2) == 0).then(None).otherwise(base)
    else:
        expr = pl.when((base % 2) == 0).then(pl.lit(-5)).otherwise(base)

    q = ldf.select(expr.rank(method=method, descending=descending))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("method", ["random"])
def test_rank_unsupported(ldf: pl.LazyFrame, method: RankMethod, seed: int) -> None:
    expr = pl.col("a").rank(method=method, seed=seed)
    q = ldf.select(expr)
    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize("mode", ["half_to_even", "half_away_from_zero"])
def test_round(ldf: pl.LazyFrame, mode: RoundMethod) -> None:
    q = ldf.select(pl.col("a").sin().round(2, mode=mode))
    assert_gpu_result_equal(q)
