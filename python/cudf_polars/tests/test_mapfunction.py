# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


def test_merge_sorted_raises():
    df1 = pl.LazyFrame({"a": [1, 6, 9], "b": [1, -10, 4]})
    df2 = pl.LazyFrame({"a": [-1, 5, 11, 20], "b": [2, 7, -4, None]})
    df3 = pl.LazyFrame({"a": [-10, 20, 21], "b": [1, 2, 3]})

    q = df1.merge_sorted(df2, key="a").merge_sorted(df3, key="a")

    assert_ir_translation_raises(q, NotImplementedError)


def test_explode_multiple_raises():
    df = pl.LazyFrame({"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]})
    q = df.explode("a", "b")

    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize("column", ["a", "b"])
def test_explode_single(column):
    df = pl.LazyFrame(
        {
            "a": [[1, 2], [3, 4], None],
            "b": [[5, 6], [7, 8], [9, 10]],
            "c": [None, 11, 12],
        }
    )
    q = df.explode(column)

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("mapping", [{"b": "a"}, {"a": "c", "b": "c"}])
def test_rename_duplicate_raises(mapping):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

    q = df.rename(mapping)

    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize(
    "mapping", [{}, {"b": "c"}, {"b": "a", "a": "b"}, {"a": "c", "b": "d"}]
)
def test_rename_columns(mapping):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

    q = df.rename(mapping)

    assert_gpu_result_equal(q)
