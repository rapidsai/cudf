# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
        # "exp", Missing rust side impl
        # "log", Missing rust side impl
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
def ldf(dtype):
    return pl.DataFrame({"a": pl.Series([1, 2, None, 4, 5], dtype=dtype)}).lazy()


def test_unary(ldf, op):
    expr = getattr(pl.col("a"), op)()
    q = ldf.select(expr)
    assert_gpu_result_equal(q, check_exact=False)
