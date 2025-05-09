# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
def test_rolling_datetime(time_unit):
    dates = [
        "2020-01-01 13:45:48",
        "2020-01-01 16:42:13",
        "2020-01-01 16:45:09",
        "2020-01-02 18:12:48",
        "2020-01-03 19:45:32",
        "2020-01-08 23:16:43",
    ]
    df = (
        pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]})
        .with_columns(pl.col("dt").str.strptime(pl.Datetime(time_unit)))
        .lazy()
    )
    q = df.with_columns(
        sum_a=pl.sum("a").rolling(index_column="dt", period="2d"),
        min_a=pl.min("a").rolling(index_column="dt", period="5d"),
        max_a=pl.max("a").rolling(index_column="dt", period="10d", offset="2d"),
    )

    assert_gpu_result_equal(q)


def test_rolling_date_raises():
    dates = [
        "2020-01-01",
        "2020-01-01",
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-08",
    ]
    df = (
        pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]})
        .with_columns(pl.col("dt").str.strptime(pl.Date()))
        .lazy()
    )
    q = df.with_columns(
        max_a=pl.max("a").rolling(index_column="dt", period="10d", offset="2d"),
    )

    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt32, pl.Int64, pl.UInt64])
def test_rolling_integral_orderby(dtype):
    df = pl.LazyFrame(
        {
            "orderby": pl.Series([1, 4, 8, 10, 12, 13, 14, 22], dtype=dtype),
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.with_columns(
        pl.col("values").sum().rolling("orderby", period="4i", closed="both")
    )

    assert_gpu_result_equal(q)


def test_rolling_collect_list_raises():
    df = pl.LazyFrame(
        {
            "orderby": [1, 4, 8, 10, 12, 13, 14, 22],
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    assert_ir_translation_raises(
        df.with_columns(pl.col("values").rolling("orderby", period="4i")),
        NotImplementedError,
    )


def test_unsorted_raises_computeerror():
    df = pl.LazyFrame({"orderby": [1, 2, 4, 2], "values": [1, 2, 3, 4]})
    q = df.select(pl.col("values").sum().rolling("orderby", period="2i"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine=pl.GPUEngine(raise_on_fail=True))


def test_orderby_nulls_raises_computeerror():
    df = pl.LazyFrame({"orderby": [1, 2, 4, None], "values": [1, 2, 3, 4]})
    q = df.select(pl.col("values").sum().rolling("orderby", period="2i"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine=pl.GPUEngine(raise_on_fail=True))


def test_grouped_rolling():
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5, 6], "b": [1, 2, 1, 3, 1, 2]})

    q = df.select(pl.col("a").min().over("b"))

    assert_ir_translation_raises(q, NotImplementedError)


def test_rolling_inside_groupby_raises():
    df = pl.LazyFrame(
        {"keys": [1, 1, 1, 2], "orderby": [1, 2, 4, 2], "values": [1, 2, 3, 4]}
    )
    q = df.group_by("keys").agg(pl.col("values").rolling("orderby", period="2i").sum())

    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")

    assert_ir_translation_raises(q, NotImplementedError)
