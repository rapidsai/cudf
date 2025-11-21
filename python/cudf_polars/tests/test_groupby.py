# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import random
from datetime import date
from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_132, POLARS_VERSION_LT_1321


@pytest.fixture
def df():
    lf = pl.LazyFrame(
        {
            "key1": [1, 1, 1, 2, 3, 1, 4, 6, 7],
            "key2": [2, 2, 2, 2, 6, 1, 4, 6, 8],
            "int": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "int32": pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=pl.Int32()),
            "uint16_with_null": pl.Series(
                [1, None, 2, None, None, None, 4, 5, 6], dtype=pl.UInt16()
            ),
            "float": [7.0, 1, 2, 3, 4.5, 5, 6, 7, 8],
            "string": ["abc", "def", "hijk", "lmno", "had", "to", "be", "or", "not"],
            "datetime": [
                date(1970, 1, 1),
                date(1972, 1, 10),
                date(2000, 1, 1),
                date(2004, 12, 1),
                date(2004, 10, 1),
                date(1971, 2, 1),
                date(2003, 12, 1),
                date(2001, 1, 1),
                date(1999, 12, 31),
            ],
        }
    )
    if not POLARS_VERSION_LT_132:
        lf = lf.with_columns(
            pl.col("float").cast(pl.Decimal(precision=9, scale=2)).alias("decimal")
        )
    return lf


@pytest.fixture(
    params=[
        [pl.col("key1")],
        [pl.col("key2")],
        [pl.col("key1"), pl.lit(1)],
        [pl.col("key1") * pl.col("key2")],
        [pl.col("key1"), pl.col("key2")],
        [pl.col("key1") == pl.col("key2")],
        [pl.col("key2"), pl.col("key1") == pl.lit(1, dtype=pl.Int64)],
    ],
    ids=lambda keys: "-".join(map(str, keys)),
)
def keys(request):
    return request.param


_EXPRS: list[list[pl.Expr | str]] = [
    [],
    ["int"],
    ["float", "int"],
    [pl.col("float") + pl.col("int")],
    [pl.col("float").is_not_null()],
    [pl.col("int32").sum()],
    [pl.col("int32").mean()],
    [
        pl.col("uint16_with_null").sum(),
        pl.col("uint16_with_null").mean().alias("mean"),
    ],
    [pl.col("float").max() - pl.col("int").min() + pl.col("int").max()],
    [pl.col("float").mean(), pl.col("int").std()],
    [(pl.col("float") - pl.lit(2)).max()],
    [pl.lit(10).alias("literal_value")],
    [pl.col("float").sum().round(decimals=1)],
    [pl.col("float").round(decimals=1).sum()],
    [pl.col("float").sum().round()],
    [pl.col("float").round().sum()],
    [pl.col("int").first(), pl.col("float").last()],
    [pl.col("int").sum(), pl.col("string").str.replace("h", "foo", literal=True)],
    [pl.col("float").quantile(0.3, interpolation="nearest")],
    [pl.col("float").quantile(0.3, interpolation="higher")],
    [pl.col("float").quantile(0.3, interpolation="lower")],
    [pl.col("float").quantile(0.3, interpolation="midpoint")],
    [pl.col("float").quantile(0.3, interpolation="linear")],
    [
        pl.col("datetime").max(),
        pl.col("datetime").max().dt.is_leap_year().alias("leapyear"),
    ],
]

# polars gives us precision=None, which we
# do not supprt
if not POLARS_VERSION_LT_132:
    _EXPRS.append([pl.col("decimal").median()])


@pytest.fixture(
    params=_EXPRS,
    ids=lambda aggs: "-".join(map(str, aggs)),
)
def exprs(request):
    return request.param


@pytest.fixture(
    params=[False, True],
    ids=["no_maintain_order", "maintain_order"],
)
def maintain_order(request):
    return request.param


def test_groupby(df: pl.LazyFrame, maintain_order, keys, exprs):
    q = df.group_by(*keys, maintain_order=maintain_order).agg(*exprs)

    if not maintain_order:
        sort_keys = list(q.collect_schema().keys())[: len(keys)]
        q = q.sort(*sort_keys)

    assert_gpu_result_equal(q, check_exact=False)


def test_groupby_sorted_keys(df: pl.LazyFrame, keys, exprs):
    sorted_keys = [
        key.sort(descending=descending)
        for key, descending in zip(keys, itertools.cycle([False, True]))
    ]

    q = df.group_by(*sorted_keys).agg(*exprs)

    schema = q.collect_schema()
    sort_keys = list(schema.keys())[: len(keys)]
    # Multiple keys don't do sorting
    qsorted = q.sort(*sort_keys)
    if len(keys) > 1:
        # https://github.com/pola-rs/polars/issues/17556
        # Can't assert that the query without post-sorting fails,
        # since it _might_ pass.
        assert_gpu_result_equal(qsorted, check_exact=False)
    elif schema[sort_keys[0]] == pl.Boolean():
        # Boolean keys don't do sorting, so we get random order
        assert_gpu_result_equal(qsorted, check_exact=False)
    else:
        assert_gpu_result_equal(q, check_exact=False)


def test_groupby_len(df, keys):
    q = df.group_by(*keys).agg(pl.len())

    assert_gpu_result_equal(q, check_row_order=False)


@pytest.mark.parametrize(
    "expr",
    [
        (pl.col("int").max() + pl.col("float").min()).max(),
        (
            pl.when((pl.col("float") - pl.col("float").mean()) > 0)
            .then(pl.col("float"))
            .otherwise(None)
            .sum()
        ),
        (pl.when(pl.col("int") > 5).then(pl.col("float")).otherwise(pl.lit(0.0))),
        (pl.when(pl.col("int").min() >= 3).then(pl.col("float"))),
    ],
)
def test_groupby_unsupported(df: pl.LazyFrame, expr: pl.Expr) -> None:
    q = df.group_by("key1").agg(expr)

    assert_ir_translation_raises(q, NotImplementedError)


def test_groupby_null_keys(maintain_order):
    df = pl.LazyFrame(
        {
            "key": pl.Series([1, float("nan"), 2, None, 2, None], dtype=pl.Float64()),
            "value": [-1, 2, 1, 2, 3, 4],
        }
    )

    q = df.group_by("key", maintain_order=maintain_order).agg(pl.col("value").min())
    if not maintain_order:
        q = q.sort("key")

    assert_gpu_result_equal(q)


@pytest.mark.xfail(reason="https://github.com/pola-rs/polars/issues/17513")
def test_groupby_minmax_with_nan():
    df = pl.LazyFrame(
        {"key": [1, 2, 2, 2], "value": [float("nan"), 1, -1, float("nan")]}
    )

    q = df.group_by("key").agg(
        pl.col("value").max().alias("max"), pl.col("value").min().alias("min")
    )

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("op", [pl.Expr.nan_max, pl.Expr.nan_min])
def test_groupby_nan_minmax_raises(op):
    df = pl.LazyFrame(
        {"key": [1, 2, 2, 2], "value": [float("nan"), 1, -1, float("nan")]}
    )

    q = df.group_by("key").agg(op(pl.col("value")))

    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize(
    "key",
    [1, pl.col("key1")],
)
@pytest.mark.parametrize(
    "expr",
    [
        pl.lit(1).alias("value"),
        pytest.param(
            pl.lit([[4, 5, 6]]).alias("value"),
            marks=pytest.mark.xfail(reason="Need to expose OtherScalar in rust IR"),
        ),
        pytest.param(
            pl.Series("value", [[4, 5, 6]], dtype=pl.List(pl.Int32)),
            marks=pytest.mark.xfail(
                condition=not POLARS_VERSION_LT_1321,
                reason="https://github.com/rapidsai/cudf/issues/19610",
            ),
        ),
        pl.col("float") * (1 - pl.col("int")),
        [pl.lit(2).alias("value"), pl.col("float") * 2],
    ],
)
def test_groupby_literal_in_agg(df, key, expr):
    # check_row_order=False doesn't work for list aggregations
    # so just sort by the group key
    q = df.group_by(key).agg(expr).sort(key, maintain_order=True)
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "expr",
    [pl.col("int").unique(), pl.col("int").drop_nulls(), pl.col("int").cum_max()],
)
def test_groupby_unary_non_pointwise_raises(df, expr):
    q = df.group_by("key1").agg(expr)
    assert_ir_translation_raises(q, NotImplementedError)


def test_groupby_agg_broadcast_raises(df):
    q = df.group_by("key1").agg(pl.col("int") + pl.col("float").max())
    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize("nrows", [30, 300, 300_000])
@pytest.mark.parametrize("nkeys", [1, 2, 4])
def test_groupby_maintain_order_random(nrows, nkeys, with_nulls):
    key_names = [f"key{key}" for key in range(nkeys)]
    rng = random.Random(2)
    key_values = [rng.choices(range(100), k=nrows) for _ in key_names]
    value = rng.choices(range(-100, 100), k=nrows)
    df = pl.DataFrame(dict(zip(key_names, key_values, strict=True), value=value))
    if with_nulls:
        df = df.with_columns(
            *(
                pl.when(pl.col(name) == 1)
                .then(None)
                .otherwise(pl.col(name))
                .alias(name)
                for name in key_names
            )
        )
    q = df.lazy().group_by(key_names, maintain_order=True).agg(pl.col("value").sum())
    # The streaming executor is too slow for large n_rows with blocksize_mode="small"
    assert_gpu_result_equal(q, blocksize_mode="default" if nrows > 30 else None)


def test_groupby_len_with_nulls():
    df = pl.DataFrame({"a": [1, 1, 1, 2], "b": [1, None, 2, 3]})
    q = df.lazy().group_by("a").agg(pl.col("b").len())
    assert_gpu_result_equal(q, check_row_order=False)


@pytest.mark.parametrize("column", ["int", "string", "uint16_with_null"])
def test_groupby_nunique(df: pl.LazyFrame, column):
    q = df.group_by("key1").agg(pl.col(column).n_unique())

    assert_gpu_result_equal(q, check_row_order=False)


def test_groupby_null_count(df: pl.LazyFrame):
    q = df.group_by("key1").agg(pl.col("uint16_with_null").null_count())

    assert_gpu_result_equal(q, check_row_order=False)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("int").all(),
        pl.col("int").any(),
        pl.col("int").is_duplicated(),
        pl.col("int").is_first_distinct(),
        pl.col("int").is_last_distinct(),
        pl.col("int").is_unique(),
    ],
    ids=[
        "all_horizontal",
        "any_horizontal",
        "is_duplicated",
        "is_first_distinct",
        "is_last_distinct",
        "is_unique",
    ],
)
def test_groupby_unsupported_non_pointwise_boolean_function(df: pl.LazyFrame, expr):
    q = df.group_by("key1").agg(expr)
    assert_ir_translation_raises(q, NotImplementedError)


def test_groupby_mean_type_promotion(df: pl.LazyFrame) -> None:
    df = df.with_columns(pl.col("float").cast(pl.Float32))

    q = df.group_by("key1").agg(pl.col("float").mean())

    assert_gpu_result_equal(q, check_row_order=False)


def test_groupby_sum_all_null_group_returns_null():
    df = pl.LazyFrame(
        {
            "key": ["a", "a", "b", "b", "c"],
            "null_groups": [None, None, None, 2, None],
        }
    )

    q = df.group_by("key").agg(out=pl.col("null_groups").sum())
    assert_gpu_result_equal(q, check_row_order=False)


@pytest.mark.parametrize(
    "agg_expr",
    [
        pl.all().sum(),
        pl.all().mean(),
        pl.all().median(),
        pl.all().quantile(0.5),
    ],
    ids=["sum", "mean", "median", "quantile-0.5"],
)
def test_groupby_aggs_keep_unsupported_as_null(df: pl.LazyFrame, agg_expr) -> None:
    lf = df.filter(pl.col("datetime") == date(2004, 12, 1))
    q = lf.group_by("datetime").agg(agg_expr)
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "expr",
    [
        pl.when(pl.col("int") > 5).then(pl.col("float")).otherwise(None).sum(),
        pl.when(pl.col("float").count() > 0)
        .then(pl.col("float").sum())
        .otherwise(None),
        (
            pl.when(pl.col("float").min() < pl.col("float").max())
            .then(pl.col("float").max() - pl.col("float").min())
            .otherwise(pl.lit(0.0))
        ),
        (
            pl.when(pl.col("int").count() > 0)
            .then(
                pl.col("int").cast(pl.Float64).sum()
                / pl.col("int").count().cast(pl.Float64)
            )
            .otherwise(None)
        ),
    ],
    ids=[
        "pre_pointwise_then_sum",
        "post_over_aggs",
        "post_multiple_aggs_range",
        "post_manually_compute_mean",
    ],
)
def test_groupby_ternary_supported(df: pl.LazyFrame, expr: pl.Expr) -> None:
    q = df.group_by("key1").agg(expr)
    assert_gpu_result_equal(q, check_row_order=False)


@pytest.mark.parametrize(
    "strategy", ["forward", "backward", "min", "max", "mean", "zero", "one"]
)
def test_groupby_fill_null_with_strategy(strategy):
    lf = pl.LazyFrame(
        {
            "key": [1, 1, 2, 2, 2],
            "val": [None, 2, None, 4, None],
        }
    )

    q = lf.group_by("key").agg(pl.col("val").fill_null(strategy=strategy))

    assert_ir_translation_raises(q, NotImplementedError)


def test_groupby_rank_raises(df: pl.LazyFrame) -> None:
    q = df.group_by("key1").agg(pl.col("int").rank())

    assert_ir_translation_raises(q, NotImplementedError)


def test_groupby_sum_decimal_null_group() -> None:
    df = pl.LazyFrame(
        {"key1": [1, 1, 2, 3], "foo": [None, None, Decimal("1.00"), Decimal("2.00")]},
        schema={"key1": pl.Int32, "foo": pl.Decimal(9, 2)},
    )
    q = df.group_by("key1").agg(pl.col("foo").sum())
    assert_gpu_result_equal(q, check_row_order=False)
