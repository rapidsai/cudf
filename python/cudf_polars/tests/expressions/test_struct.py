# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture
def ldf():
    return pl.LazyFrame({"a": [{"b": "c", "d": "e"}, {"b": None, "d": "g"}]})


@pytest.mark.parametrize("name", [None, "my_count"])
@pytest.mark.parametrize("normalize", [True, False])
def test_value_counts(ldf, name, normalize):
    # sort=True since order is non-deterministic
    query = ldf.select(
        pl.col("a").value_counts(sort=True, name=name, normalize=normalize)
    )
    assert_gpu_result_equal(query)


def test_value_counts_normalize_div_by_zero():
    ldf = pl.LazyFrame({"a": []}, schema={"a": pl.Int64()})
    query = ldf.select(pl.col("a").value_counts(normalize=True))
    assert_gpu_result_equal(query)


def test_groupby_value_counts_notimplemented():
    lgb = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]}).group_by("a")
    value_counts_expr = pl.col("b").value_counts()
    query = lgb.agg(value_counts_expr)
    assert_ir_translation_raises(query, NotImplementedError)

    query = lgb.agg(value_counts_expr.first())
    assert_ir_translation_raises(query, NotImplementedError)


def test_struct(ldf):
    query = ldf.select(pl.struct(pl.all()))
    assert_gpu_result_equal(query)


def test_nested_struct():
    ldf = pl.LazyFrame({"a": [{"x": {"i": 0, "j": 0}, "y": {"i": 0, "k": 1}}]})
    query = ldf.select(pl.struct(pl.all()))
    assert_gpu_result_equal(query)
