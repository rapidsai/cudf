# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
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


def test_field_getitem(ldf):
    query = ldf.select(pl.col("a").struct[0])
    assert_gpu_result_equal(query)


def test_field(ldf):
    query = ldf.select(pl.col("a").struct.field("b"))
    assert_gpu_result_equal(query)


def test_unnest(ldf):
    query = ldf.select(pl.col("a").struct.unnest())
    assert_gpu_result_equal(query)


def test_json_encode(ldf):
    query = ldf.select(pl.col("a").struct.json_encode())
    assert_gpu_result_equal(query)


def test_rename_fields(ldf):
    query = ldf.select(pl.col("a").struct.rename_fields(["1", "2"]).struct.unnest())
    assert_gpu_result_equal(query)


def test_with_fields(ldf):
    query = ldf.select(
        pl.col("a").struct.with_fields(pl.field("b").str.len_chars()).struct.unnest()
    )
    assert_ir_translation_raises(query, NotImplementedError)


@pytest.mark.parametrize("method", ["prefix_fields", "suffix_fields"])
def test_prefix_suffix_fields(ldf, method):
    query = ldf.select(getattr(pl.col("a").name, method)("foo").struct.unnest())
    assert_gpu_result_equal(query)


def test_map_field_names(ldf):
    query = ldf.select(pl.col("a").name.map_fields(lambda x: x.upper()).struct.unnest())
    assert_ir_translation_raises(query, NotImplementedError)
