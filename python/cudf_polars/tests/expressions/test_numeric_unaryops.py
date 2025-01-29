# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


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
        iinfo = np.iinfo("int32")
        values.append(iinfo.min)
        values.append(iinfo.max)
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
