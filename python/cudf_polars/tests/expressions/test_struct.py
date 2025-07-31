# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_131


@pytest.fixture
def ldf():
    return pl.LazyFrame(
        {"a": [{"b": "c", "d": "e", "f": "g"}, {"b": None, "d": "g", "f": "h"}]}
    )


def test_field_getitem(request, ldf):
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_131,
            reason="not supported until polars 1.31",
        )
    )
    q = ldf.select(pl.col("a").struct[0])
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("fields", [("b",), ("b", "d"), ("^b.*|f.*$",)])
def test_field(request, ldf, fields):
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_131,
            reason="not supported until polars 1.31",
        )
    )
    q = ldf.select(pl.col("a").struct.field(*fields))
    assert_gpu_result_equal(q)


def test_unnest(request, ldf):
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_131,
            reason="not supported until polars 1.31",
        )
    )
    q = ldf.select(pl.col("a").struct.unnest())
    assert_gpu_result_equal(q)


def test_json_encode(request, ldf):
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_131,
            reason="not supported until polars 1.31",
        )
    )
    q = ldf.select(pl.col("a").struct.json_encode())
    assert_gpu_result_equal(q)

    ldf_newlines = pl.LazyFrame({"a": [{"b": "c\nd", "d": "\r\nz"}]})
    q = ldf_newlines.select(pl.col("a").struct.json_encode())
    assert_gpu_result_equal(q)


def test_rename_fields(request, ldf):
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_131,
            reason="not supported until polars 1.31",
        )
    )
    q = ldf.select(pl.col("a").struct.rename_fields(["1", "2", "3"]).struct.unnest())
    assert_gpu_result_equal(q)


def test_with_fields(ldf):
    q = ldf.select(
        pl.col("a").struct.with_fields(pl.field("b").str.len_chars()).struct.unnest()
    )
    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize(
    "expr",
    [pl.col("a").name.prefix_fields, pl.col("a").name.suffix_fields],
    ids=lambda x: x.__name__,
)
def test_prefix_suffix_fields(request, ldf, expr):
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_131,
            reason="not supported until polars 1.31",
        )
    )
    q = ldf.select(expr("foo").struct.unnest())
    assert_gpu_result_equal(q)


def test_map_field_names(ldf):
    q = ldf.select(pl.col("a").name.map_fields(lambda x: x.upper()).struct.unnest())
    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize("name", [None, "my_count"])
@pytest.mark.parametrize("normalize", [True, False])
def test_value_counts(ldf, name, normalize):
    # sort=True since order is non-deterministic
    q = ldf.select(pl.col("a").value_counts(sort=True, name=name, normalize=normalize))
    assert_gpu_result_equal(q)


def test_value_counts_normalize_div_by_zero():
    ldf = pl.LazyFrame({"a": []}, schema={"a": pl.Int64()})
    q = ldf.select(pl.col("a").value_counts(normalize=True))
    assert_gpu_result_equal(q)


def test_groupby_value_counts_notimplemented():
    lgb = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]}).group_by("a")
    value_counts_expr = pl.col("b").value_counts()
    q = lgb.agg(value_counts_expr)
    assert_ir_translation_raises(q, NotImplementedError)

    q = lgb.agg(value_counts_expr.first())
    assert_ir_translation_raises(q, NotImplementedError)


def test_struct(ldf):
    q = ldf.select(pl.struct(pl.all()))
    assert_gpu_result_equal(q)


def test_nested_struct():
    ldf = pl.LazyFrame({"a": [{"x": {"i": 0, "j": 0}, "y": {"i": 0, "k": 1}}]})
    q = ldf.select(pl.struct(pl.all()))
    assert_gpu_result_equal(q)


def test_value_counts_with_nulls(ldf):
    ldf_with_nulls = ldf.select(c=pl.Series(["x", None, "y", "x", None, "x"]))
    q = ldf_with_nulls.select(pl.col("c").value_counts(sort=True))
    assert_gpu_result_equal(q)
