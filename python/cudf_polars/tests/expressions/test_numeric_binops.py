# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_132

dtypes = [
    pl.Int8,
    pl.Int16,
    pl.Int64,
    pl.UInt8,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
]


@pytest.fixture(params=dtypes)
def ltype(request):
    return request.param


@pytest.fixture(params=dtypes)
def rtype(request):
    return request.param


@pytest.fixture(
    params=[
        pl.Expr.eq,
        pl.Expr.eq_missing,
        pl.Expr.ne,
        pl.Expr.ne_missing,
        pl.Expr.lt,
        pl.Expr.le,
        pl.Expr.gt,
        pl.Expr.ge,
        pl.Expr.add,
        pl.Expr.sub,
        pl.Expr.mul,
        pl.Expr.truediv,
        pl.Expr.floordiv,
        pl.Expr.mod,
    ],
    ids=lambda fn: fn.__name__,
)
def binop(request):
    return request.param


@pytest.fixture
def df(request, ltype, rtype, with_nulls, binop):
    a = [1, 2, 3, 5, 8]
    if with_nulls:
        a[2] = None
        a[-1] = None
    b = [10, 20, 30, 50, 0]
    if with_nulls:
        b[1] = None
        b[3] = None
        b[-1] = None

    return pl.LazyFrame({"a": a, "b": b}, schema={"a": ltype, "b": rtype})


def test_numeric_binop(df, binop):
    left = pl.col("a")
    right = pl.col("b")

    q = df.select(binop(left, right))

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("left_scalar", [False, True])
@pytest.mark.parametrize("right_scalar", [False, True])
def test_binop_with_scalar(left_scalar, right_scalar):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [5, 6, 7]})

    lop = pl.lit(2) if left_scalar else pl.col("a")
    rop = pl.lit(6) if right_scalar else pl.col("b")
    q = df.select(lop / rop)

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("zero", [0, pl.lit(0)])
def test_floor_div_binop_by_zero(zero, ltype):
    df = pl.LazyFrame({"a": [1, 0, 3]}, schema={"a": ltype})

    q = df.select(pl.col("a") // zero)

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("divisor", [1, 2.0])
def test_true_div_boolean_column(divisor):
    df = pl.LazyFrame({"a": [True, False]})

    q = df.select(pl.col("a") / divisor)

    assert_gpu_result_equal(q)


def test_true_div_with_decimals():
    df = pl.LazyFrame(
        {
            "foo": [Decimal("1.00"), Decimal("2.00"), Decimal("3.00"), None],
            "bar": [Decimal("4.00"), Decimal("5.00"), Decimal("6.00"), Decimal("1.00")],
        },
        schema={"foo": pl.Decimal(15, 2), "bar": pl.Decimal(15, 2)},
    )
    q = df.select(pl.col("bar") / pl.col("foo"))
    assert_gpu_result_equal(q, check_dtypes=not POLARS_VERSION_LT_132)
