# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.dsl import expr
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(params=sorted(expr.Agg._SUPPORTED))
def agg(request):
    return request.param


@pytest.fixture(params=[pl.Int32, pl.Float32, pl.Int16])
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


@pytest.mark.parametrize(
    "propagate_nans",
    [pytest.param(False, marks=pytest.mark.xfail(reason="Need to mask nans")), True],
    ids=["mask_nans", "propagate_nans"],
)
@pytest.mark.parametrize("op", ["min", "max"])
def test_agg_float_with_nans(propagate_nans, op):
    df = pl.LazyFrame({"a": pl.Series([1, 2, float("nan")], dtype=pl.Float64())})
    op = getattr(pl.Expr, f"nan_{op}" if propagate_nans else op)
    q = df.select(op(pl.col("a")))

    assert_gpu_result_equal(q)
