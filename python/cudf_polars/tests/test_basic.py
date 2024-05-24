# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def ldf_datetime():
    dates = [
        "2020-01-01 13:45:48",
        "2020-01-01 16:42:13",
        "2020-01-01 16:45:09",
        "2020-01-02 18:12:48",
        "2020-01-03 19:45:32",
        "2020-01-08 23:16:43",
    ]
    return (
        pl.DataFrame({"dt": dates, "a": [1, 2, 3, 4, 5, 6], "b": [1, 1, 1, 2, 2, 2]})
        .with_columns(pl.col("dt").str.strptime(pl.Datetime).set_sorted())
        .lazy()
    )


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "int_key1": np.repeat(np.arange(10), 10),
            "int_key2": np.tile(np.arange(10), 10),
            "str_key1": np.repeat(list("ABCDEFGHIJ"), 10),
            "int_val": np.random.randint(100, size=100),
            "float_val": np.random.rand(100),
        }
    )


@pytest.fixture
def ldf(df):
    return df.lazy()


@pytest.mark.xfail(reason="Rolling window not yet implemented")
def test_rolling(ldf_datetime):
    out = ldf_datetime.rolling(index_column="dt", period="2d").agg(
        [
            pl.sum("a").alias("sum_a"),
            pl.min("a").alias("min_a"),
            pl.max("a").alias("max_a"),
        ]
    )
    assert_gpu_result_equal(out)


@pytest.mark.xfail(reason="Grouped rolling window not yet implemented")
def test_groupby_rolling(ldf_datetime):
    out = ldf_datetime.rolling(index_column="dt", period="2d", group_by="b").agg(
        [
            pl.sum("a").alias("sum_a"),
            pl.min("a").alias("min_a"),
            pl.max("a").alias("max_a"),
        ]
    )
    assert_gpu_result_equal(out)


@pytest.mark.xfail(reason="Rolling expression not yet implemented")
def test_rolling_expression(ldf_datetime):
    out = ldf_datetime.with_columns(
        sum_a=pl.sum("a").rolling(index_column="dt", period="2d"),
        min_a=pl.min("a").rolling(index_column="dt", period="2d"),
        max_a=pl.max("a").rolling(index_column="dt", period="2d"),
    )
    assert_gpu_result_equal(out)


def test_datetime_comparison(ldf_datetime):
    out = ldf_datetime.filter(
        pl.col("dt") > datetime.fromisoformat("2020-01-01 16:45:09")
    )
    assert_gpu_result_equal(out)


@pytest.fixture
def null_data():
    return pl.DataFrame(
        {
            "a": [1, 2, None, 4, None],
        }
    ).lazy()


def test_drop_nulls(null_data):
    result = null_data.drop_nulls()
    assert_gpu_result_equal(result)
