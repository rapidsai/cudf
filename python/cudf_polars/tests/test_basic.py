# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import operator
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


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_binaryops(op, dtype):
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
        }
    ).lazy()

    dtype = pl.datatypes.numpy_char_code_to_dtype(dtype)
    df = df.with_columns(pl.col("a").cast(dtype)).with_columns(pl.col("b").cast(dtype))
    result = df.with_columns(op(pl.col("a"), pl.col("b")))
    assert_gpu_result_equal(result)


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


@pytest.mark.parametrize("how", ["inner", "left", "semi", "outer_coalesce"])
def test_join(df: pl.DataFrame, how):
    pl.set_random_seed(42)
    # Sample eagerly since we haven't implemented it yet.
    ldf1 = df.sample(n=50).lazy()
    ldf2 = df.sample(n=50).lazy()

    out = ldf1.join(ldf2, on=["int_key1", "int_key2"], how=how)
    assert_gpu_result_equal(out, check_row_order=False)


def test_sort(ldf):
    for col in ldf.columns:
        out = ldf.sort(by=col)
        assert_gpu_result_equal(out)


def test_filter(ldf):
    out = ldf.filter(pl.col("int_key1") > pl.col("int_key2"))
    assert_gpu_result_equal(out)


@pytest.mark.parametrize(
    "agg",
    [
        "sum",
        "min",
        "max",
        "mean",
        # TODO: first/last get turned into slice of the Scan
        "first",
        "last",
        "count",
        "median",
    ],
)
def test_agg(df, agg):
    ldf = (
        df.cast(
            {key: pl.Float64 for key in df.columns if ("int" in key or "float" in key)}
        )
        .select(list(filter(lambda c: "str" not in c, df.columns)))
        .lazy()
    )
    out = getattr(ldf, agg)()
    assert_gpu_result_equal(out, check_dtype=agg != "count", check_exact=False)


@pytest.mark.parametrize("keep", ["first", "last", "none"])
@pytest.mark.parametrize("subset", [None, "keys"])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("maintain_order", [False, True])
def test_unique(ldf: pl.LazyFrame, keep, subset, sort, maintain_order):
    if subset is not None:
        subset = list(filter(lambda c: "key" in c, ldf.columns))
        sort_by = subset
    else:
        sort_by = ldf.columns
    if sort:
        ldf = ldf.sort(*sort_by)
    out = ldf.unique(
        subset,
        keep=keep,
        maintain_order=maintain_order,
    )
    assert_gpu_result_equal(out, check_row_order=maintain_order)


def test_selection(ldf: pl.LazyFrame):
    k = pl.col("int_key1")
    v = pl.col("int_val")
    # groupby stops predicate pushdown
    out = ldf.group_by(k).agg(v.sum()).filter(k * 2 > v)
    assert_gpu_result_equal(out)


@pytest.mark.xfail(reason="arg_where not yet implemented")
def test_expr_function(ldf):
    out = ldf.select(pl.arg_where(pl.col("int_key1") == 5)).set_sorted(
        pl.col("int_key1")
    )
    assert_gpu_result_equal(out)


def test_filter_expr(ldf):
    out = ldf.select(pl.col("int_key1").filter(pl.col("int_key2") > 4))
    assert_gpu_result_equal(out)


def test_gather_expr(ldf):
    out = ldf.select(pl.col("int_key1").gather(pl.col("int_key2")))
    assert_gpu_result_equal(out)
