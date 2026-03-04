# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_132


@pytest.fixture(
    params=[
        [1, 2, 1, 3, 5, None, None],
        [1.5, 2.5, None, 1.5, 3, float("nan"), 3],
        [],
        [None, None],
        [1, 2, 3, 4, 5],
    ]
)
def null_data(request):
    is_empty = pl.Series(request.param).dtype == pl.Null
    return pl.DataFrame(
        {
            "a": pl.Series(request.param, dtype=pl.Float64 if is_empty else None),
            "b": pl.Series(request.param, dtype=pl.Float64 if is_empty else None),
        }
    ).lazy()


def test_drop_null(null_data):
    q = null_data.select(pl.col("a").drop_nulls())
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "value",
    [0, pl.col("a").mean(), pl.col("b")],
    ids=["scalar", "aggregation", "column_expression"],
)
def test_fill_null(null_data, value):
    q = null_data.select(pl.col("a").fill_null(value))
    assert_gpu_result_equal(q)


def test_fill_null_with_string():
    q = pl.LazyFrame({"a": [None, "a"]}).select(pl.col("a").fill_null("b"))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "strategy", ["forward", "backward", "min", "max", "mean", "zero", "one"]
)
def test_fill_null_with_strategy(null_data, strategy):
    q = null_data.select(pl.col("a").fill_null(strategy=strategy))
    if POLARS_VERSION_LT_132:
        assert_ir_translation_raises(q, NotImplementedError)
    else:
        assert_gpu_result_equal(q)


@pytest.mark.parametrize("strategy", ["zero", "one"])
def test_fill_null_with_strategy_bool(strategy):
    q = pl.LazyFrame({"a": [True, None, False]}).select(
        pl.col("a").fill_null(strategy=strategy)
    )
    if POLARS_VERSION_LT_132:
        assert_ir_translation_raises(q, NotImplementedError)
    else:
        assert_gpu_result_equal(q)


@pytest.mark.parametrize("strategy", ["forward", "backward"])
@pytest.mark.parametrize("limit", [0, 1, 2])
def test_fill_null_with_limit(null_data, strategy, limit):
    q = null_data.select(pl.col("a").fill_null(strategy=strategy, limit=limit))
    if limit != 0:
        assert_ir_translation_raises(q, NotImplementedError)
    else:
        if POLARS_VERSION_LT_132:
            assert_ir_translation_raises(q, NotImplementedError)
        else:
            assert_gpu_result_equal(q)
