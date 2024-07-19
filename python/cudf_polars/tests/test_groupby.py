# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils import versions


@pytest.fixture
def df():
    return pl.LazyFrame(
        {
            "key1": [1, 1, 1, 2, 3, 1, 4, 6, 7],
            "key2": [2, 2, 2, 2, 6, 1, 4, 6, 8],
            "int": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "float": [7.0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )


@pytest.fixture(
    params=[
        [pl.col("key1")],
        [pl.col("key2")],
        [pl.col("key1") * pl.col("key2")],
        [pl.col("key1"), pl.col("key2")],
        [pl.col("key1") == pl.col("key2")],
        [pl.col("key2"), pl.col("key1") == pl.lit(1, dtype=pl.Int64)],
    ],
    ids=lambda keys: "-".join(map(str, keys)),
)
def keys(request):
    return request.param


@pytest.fixture(
    params=[
        ["int"],
        ["float", "int"],
        [pl.col("float") + pl.col("int")],
        [pl.col("float").max() - pl.col("int").min()],
        [pl.col("float").mean(), pl.col("int").std()],
        [(pl.col("float") - pl.lit(2)).max()],
        [pl.col("float").sum().round(decimals=1)],
        [pl.col("float").round(decimals=1).sum()],
    ],
    ids=lambda aggs: "-".join(map(str, aggs)),
)
def exprs(request):
    return request.param


@pytest.fixture(
    params=[
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="Maintaining order in groupby not implemented"
            ),
        ),
    ],
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
        with pytest.raises(AssertionError):
            # https://github.com/pola-rs/polars/issues/17556
            assert_gpu_result_equal(q, check_exact=False)
        if versions.POLARS_VERSION_LT_12 and schema[sort_keys[1]] == pl.Boolean():
            # https://github.com/pola-rs/polars/issues/17557
            with pytest.raises(AssertionError):
                assert_gpu_result_equal(qsorted, check_exact=False)
        else:
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
        pl.col("float").is_not_null(),
        (pl.col("int").max() + pl.col("float").min()).max(),
    ],
)
def test_groupby_unsupported(df, expr):
    q = df.group_by("key1").agg(expr)

    assert_ir_translation_raises(q, NotImplementedError)


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


@pytest.mark.parametrize("key", [1, pl.col("key1")])
@pytest.mark.parametrize(
    "expr",
    [
        pl.lit(1).alias("value"),
        pl.lit([[4, 5, 6]]).alias("value"),
        pl.col("float") * (1 - pl.col("int")),
        [pl.lit(2).alias("value"), pl.col("float") * 2],
    ],
)
def test_groupby_literal_in_agg(df, key, expr):
    # check_row_order=False doesn't work for list aggregations
    # so just sort by the group key
    q = df.group_by(key).agg(expr).sort(key, maintain_order=True)
    assert_gpu_result_equal(q)
