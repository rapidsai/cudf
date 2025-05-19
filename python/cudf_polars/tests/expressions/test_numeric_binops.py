# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
)

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

    lkind = (
        "i"
        if ltype.is_signed_integer()
        else ("u" if ltype.is_unsigned_integer() else "f")
    )
    rkind = (
        "i"
        if rtype.is_signed_integer()
        else ("u" if rtype.is_unsigned_integer() else "f")
    )
    if (
        not with_nulls
        and binop.__name__ in {"floordiv", "mod"}
        # This catches the case where the result is not promoted to float.
        and (
            (lkind == rkind and lkind in {"i", "u"})
            or ({lkind, rkind} == {"i", "u"} and pl.UInt64 not in {ltype, rtype})
        )
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="Polars nullifies division by zero for integral types"
            )
        )

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
