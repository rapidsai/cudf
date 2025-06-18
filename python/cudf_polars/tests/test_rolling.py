# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_130


@pytest.fixture(params=[False, True], ids=["nosort", "sort"])
def presort(request):
    return request.param


@pytest.fixture
def df(presort):
    dates = pl.Series(
        [
            datetime.strptime("2020-01-01 13:45:48", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-01 16:42:13", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-01 16:45:09", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-02 18:12:48", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-03 19:45:32", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-08 23:16:43", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-10 23:16:43", "%Y-%m-%d %H:%M:%S"),
        ],
        dtype=pl.Datetime(time_unit="us"),
    )
    df = pl.LazyFrame(
        {
            "dt": dates,
            "values": [3, 7, 5, 9, 2, 1, 72],
            "floats": pl.Series(
                [float("nan"), 7, 5, 2, -10, 1, float("inf")], dtype=pl.Float64()
            ),
        }
    )
    if presort:
        return df.with_columns(pl.col("dt").set_sorted())
    return df


@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
@pytest.mark.parametrize("period", ["1w4d", "48h", "180s"])
def test_datetime_rolling(df, closed, period):
    q = df.rolling("dt", period=period, closed=closed).agg(
        sum_a=pl.sum("values"),
        min_a=pl.min("values"),
        max_a=pl.max("values"),
        count=pl.len(),
    )

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt32, pl.Int64, pl.UInt64])
def test_rolling_integral_orderby(dtype):
    df = pl.LazyFrame(
        {
            "orderby": pl.Series([1, 4, 8, 10, 12, 13, 14, 22], dtype=dtype),
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.rolling("orderby", period="4i", closed="both").agg(pl.col("values").sum())

    assert_gpu_result_equal(q)


def test_rolling_collect_list_raises():
    df = pl.LazyFrame(
        {
            "orderby": [1, 4, 8, 10, 12, 13, 14, 22],
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    assert_ir_translation_raises(
        df.rolling("orderby", period="4i").agg(pl.col("values")),
        NotImplementedError,
    )


@pytest.mark.parametrize("with_slice", [False, True])
def test_rolling_empty_aggs(with_slice):
    df = pl.LazyFrame(
        {
            "orderby": [1, 4, 8, 10, 12, 13, 14, 22],
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.rolling("orderby", period="2i", closed="none").agg()
    if with_slice:
        q = q.slice(2)

    assert_gpu_result_equal(q)


def test_calendrical_period_unsupported(df):
    q = df.rolling("dt", period="1mo", closed="right").agg(sum=pl.sum("values"))

    assert_ir_translation_raises(q, NotImplementedError)


def test_unsorted_raises():
    df = pl.LazyFrame({"orderby": [1, 2, 4, 2], "values": [1, 2, 3, 4]})
    q = df.rolling("orderby", period="2i").agg(sum=pl.sum("values"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    if POLARS_VERSION_LT_130:
        with pytest.raises(pl.exceptions.ComputeError):
            q.collect(engine=pl.GPUEngine(raise_on_fail=True))
    else:
        with pytest.raises(
            RuntimeError, match=r".*rolling is not sorted, please sort first"
        ):
            q.collect(engine=pl.GPUEngine(raise_on_fail=True))


def test_grouped_rolling():
    df = pl.LazyFrame(
        {
            "keys": [1, None, 2, 1, 2, None],
            "orderby": [10, 2, -11, 11, -5, 3],
            "values": [1, 2, 3, 4, 5, 6],
        }
    )
    q = df.rolling("orderby", period="5i", group_by="keys").agg(pl.col("values").sum())

    assert_gpu_result_equal(q)


def test_grouped_rolling_unsorted_raises():
    df = pl.LazyFrame(
        {
            "keys": [1, None, 2, 1, 2, None],
            "orderby": [10, 2, -11, 11, -5, -2],
            "values": [1, 2, 3, 4, 5, 6],
        }
    )
    q = df.rolling("orderby", period="5i", group_by="keys").agg(pl.col("values").sum())

    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine="in-memory")
    if POLARS_VERSION_LT_130:
        with pytest.raises(pl.exceptions.ComputeError):
            q.collect(engine=pl.GPUEngine(raise_on_fail=True))
    else:
        with pytest.raises(
            RuntimeError, match="Input for grouped rolling is not sorted"
        ):
            q.collect(engine=pl.GPUEngine(raise_on_fail=True))


def test_orderby_nulls_raises_computeerror():
    df = pl.LazyFrame({"orderby": [1, 2, 4, None], "values": [1, 2, 3, 4]})
    q = df.rolling("orderby", period="2i").agg(sum=pl.sum("values"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    if POLARS_VERSION_LT_130:
        with pytest.raises(pl.exceptions.ComputeError):
            q.collect(engine=pl.GPUEngine(raise_on_fail=True))
    else:
        with pytest.raises(
            RuntimeError, match=r"Index column.*in rolling may not contain nulls"
        ):
            q.collect(engine=pl.GPUEngine(raise_on_fail=True))


def test_rolling_nested_raises():
    q = (
        pl.LazyFrame(
            {
                "orderby": [1, 2, 3, 4, 10],
                "values": [6, 7, 8, 9, 10],
            }
        )
        .rolling("orderby", period="3i", closed="left")
        .agg(pl.col("values").rolling("orderby", period="2i").sum())
    )
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    assert_ir_translation_raises(q, NotImplementedError)


def test_unsupported_agg():
    q = (
        pl.LazyFrame(
            {
                "orderby": [1, 2, 3, 4, 10],
                "values": [6, 7, 8, 9, 10],
            }
        )
        .rolling("orderby", period="3i", closed="left")
        .agg(pl.col("values").n_unique())
    )
    assert_ir_translation_raises(q, NotImplementedError)
