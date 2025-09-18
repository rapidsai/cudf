# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_130, POLARS_VERSION_LT_132


@pytest.fixture
def df():
    return pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2],
            "x": [1, 2, 3, 4, 5],
            "x2": [1, 100, 3, 4, 50],
            "g2": ["a", "a", "b", "a", "a"],
            "g_null": [1, None, 1, None, 2],
        }
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


def test_rolling_date():
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

    assert_gpu_result_equal(q)


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


def test_unsorted_raises():
    df = pl.LazyFrame({"orderby": [1, 2, 4, 2], "values": [1, 2, 3, 4]})
    q = df.select(pl.col("values").sum().rolling("orderby", period="2i"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    if POLARS_VERSION_LT_130:
        with pytest.raises(pl.exceptions.ComputeError):
            q.collect(engine=pl.GPUEngine(raise_on_fail=True))
    else:
        with pytest.raises(
            RuntimeError,
            match=r"Index column.*in rolling is not sorted, please sort first",
        ):
            q.collect(engine=pl.GPUEngine(raise_on_fail=True))


def test_orderby_nulls_raises_computeerror():
    df = pl.LazyFrame({"orderby": [1, 2, 4, None], "values": [1, 2, 3, 4]})
    q = df.select(pl.col("values").sum().rolling("orderby", period="2i"))
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


def test_invalid_duration_spec_raises_in_translation():
    df = pl.LazyFrame({"orderby": [1, 2, 4, 5], "values": [1, 2, 3, 4]})
    q = df.select(pl.col("values").sum().rolling("orderby", period="3d"))
    assert_ir_translation_raises(q, pl.exceptions.InvalidOperationError)


def test_rolling_inside_groupby_raises():
    df = pl.LazyFrame(
        {"keys": [1, 1, 1, 2], "orderby": [1, 2, 4, 2], "values": [1, 2, 3, 4]}
    )
    q = df.group_by("keys").agg(pl.col("values").rolling("orderby", period="2i").sum())

    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")

    assert_ir_translation_raises(q, NotImplementedError)


def test_rolling_sum_all_null_window_returns_null():
    df = pl.LazyFrame(
        {
            "orderby": [1, 2, 3, 4, 5, 6],
            "null_windows": [None, None, 5, None, None, 1],
        }
    )
    q = df.select(
        out=pl.col("null_windows").sum().rolling("orderby", period="2i", closed="both")
    )
    # Expected: [null, null, 5, 5, 5, 1]
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.len().over("g"),
        pl.col("x").cast(pl.Float64).mean().round(1).over("g"),
        pl.col("x2").quantile(0.5, interpolation="lower").over("g"),
        pl.col("x").sum().over("g", "g2"),
        pl.col("x").sum().over(pl.col("g") % 2),
        pl.col("x").sum().over("g_null"),
        pl.col("x").cast(pl.Float32).mean().over("g"),
        pl.col("x").sum().over(pl.lit(1)),
    ],
    ids=[
        "sum_broadcast",
        "len_broadcast",
        "mean_round",
        "quantile_lower",
        "multi_key_partition",
        "expr_partition",
        "null_keys",
        "mean_float32_promotion",
        "literal_partition",
    ],
)
def test_over_group_various(df, expr):
    q = df.select(expr)
    assert_gpu_result_equal(q)


def test_window_over_group_sum_all_null_group_is_zero(df):
    q = df.with_columns(
        pl.when(pl.col("g") == 1)
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col("x"))
        .alias("null")
    ).select(s=pl.col("null").sum().over("g"))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "order_by",
    [
        "x",
        pl.col("x") * 2,
        pl.when((pl.col("x") % 2) == 0).then(pl.col("x")).otherwise(-pl.col("x")),
        ["x", "x2"],
        ["g_null", "g2", "x2"],
        [pl.col("g") + 7, (pl.col("x") * 3) - 2],
    ],
)
@pytest.mark.parametrize("order_by_descending", [False, True])
@pytest.mark.parametrize("order_by_nulls_last", [False, True])
def test_over_with_order_by(df, order_by, order_by_descending, order_by_nulls_last):
    q = df.select(
        pl.col("x")
        .sum()
        .over(
            "g",
            order_by=order_by,
            descending=order_by_descending,
            nulls_last=order_by_nulls_last,
        )
    )
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("strategy", ["explode", "join"], ids=["explode", "join"])
def test_over_with_mapping_strategy_unsupported(df, strategy):
    q = df.select(pl.col("x").sum().over("g", mapping_strategy=strategy))
    assert_ir_translation_raises(q, NotImplementedError)


def test_over_boolean_function_unsupported(df):
    q = df.select(pl.col("x").not_().over("g"))
    assert_ir_translation_raises(q, NotImplementedError)


def test_over_ternary(df):
    q = df.select(
        pl.when(pl.col("g") == 1)
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col("x"))
        .sum()
        .over("g")
    )

    assert_gpu_result_equal(q)


def test_over_broadcast_input_row_group_indices_aligned():
    num_rows, num_groups = 512, 64

    df = pl.LazyFrame(
        {
            "g": [(i * 31) % num_groups for i in range(num_rows)],
            "x": list(range(num_rows)),
        }
    )
    q = df.select(pl.col("x").sum().over("g"))

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("order_by", [None, ["g2", pl.col("x2") * 2]])
def test_rank_over(
    request,
    df: pl.LazyFrame,
    method: str,
    *,
    descending: bool,
    order_by: None | list[str | pl.Expr],
) -> None:
    request.applymarker(
        pytest.mark.xfail(condition=POLARS_VERSION_LT_132, reason="rank unsupported")
    )
    q = df.select(
        pl.col("x")
        .rank(method=method, descending=descending)
        .over("g", order_by=order_by)
    )
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("order_by", [None, ["g2", pl.col("x2") * 2]])
def test_rank_over_with_ties(
    request,
    df: pl.LazyFrame,
    method: str,
    *,
    descending: bool,
    order_by: None | list[str | pl.Expr],
) -> None:
    request.applymarker(
        pytest.mark.xfail(condition=POLARS_VERSION_LT_132, reason="rank unsupported")
    )
    q = df.select(
        pl.when(pl.col("g") == 2)
        .then(pl.lit(4))
        .otherwise(pl.col("x"))
        .rank(method=method, descending=descending)
        .over("g", order_by=order_by)
    )
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("order_by", [None, ["g2", pl.col("x2") * 2]])
def test_rank_over_with_null_values(
    request,
    df: pl.LazyFrame,
    method: str,
    *,
    descending: bool,
    order_by: None | list[str | pl.Expr],
) -> None:
    request.applymarker(
        pytest.mark.xfail(condition=POLARS_VERSION_LT_132, reason="rank unsupported")
    )
    q = df.select(
        pl.when((pl.col("x") % 2) == 0)
        .then(None)
        .otherwise(pl.col("x"))
        .rank(method=method, descending=descending)
        .over("g", order_by=order_by)
    )
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("order_by", [None, ["g2", pl.col("x2") * 2]])
def test_rank_over_with_null_group_keys(
    request,
    df: pl.LazyFrame,
    method: str,
    *,
    descending: bool,
    order_by: None | list[str | pl.Expr],
) -> None:
    request.applymarker(
        pytest.mark.xfail(condition=POLARS_VERSION_LT_132, reason="rank unsupported")
    )
    q = df.select(
        pl.col("x")
        .rank(method=method, descending=descending)
        .over("g_null", order_by=order_by)
    )
    assert_gpu_result_equal(q)
