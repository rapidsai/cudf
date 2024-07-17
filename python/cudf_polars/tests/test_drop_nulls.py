# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


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
        {"a": pl.Series(request.param, dtype=pl.Float64 if is_empty else None)}
    ).lazy()


def test_drop_null(null_data):
    q = null_data.select(pl.col("a").drop_nulls())
    assert_gpu_result_equal(q)


def test_fill_null(null_data):
    q = null_data.select(pl.col("a").fill_null(0))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "strategy", ["forward", "backward", "min", "max", "mean", "zero", "one"]
)
def test_fill_null_with_strategy(null_data, strategy):
    q = null_data.select(pl.col("a").fill_null(strategy=strategy))
    assert_ir_translation_raises(q, NotImplementedError)
