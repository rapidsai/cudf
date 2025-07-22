# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.dsl.translate import Translator
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_131


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

    if POLARS_VERSION_LT_131:
        assert_ir_translation_raises(q, NotImplementedError)
    else:
        # Now raises before translation
        with pytest.raises(pl.exceptions.DuplicateError, match="is duplicate"):
            assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize(
    "mapping", [{}, {"b": "c"}, {"b": "a", "a": "b"}, {"a": "c", "b": "d"}]
)
def test_rename_columns(mapping):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

    q = df.rename(mapping)

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("index", [None, ["a"], ["d", "a"]])
@pytest.mark.parametrize("variable_name", [None, "names"])
@pytest.mark.parametrize("value_name", [None, "unpivoted"])
def test_unpivot(index, variable_name, value_name):
    df = pl.LazyFrame(
        {
            "a": ["x", "y", "z"],
            "b": pl.Series([1, 3, 5], dtype=pl.Int16),
            "c": pl.Series([2, 4, 6], dtype=pl.Float32),
            "d": ["a", "b", "c"],
        }
    )
    q = df.unpivot(
        ["c", "b"], index=index, variable_name=variable_name, value_name=value_name
    )

    assert_gpu_result_equal(q)


def test_unpivot_defaults():
    df = pl.LazyFrame(
        {
            "a": pl.Series([11, 12, 13], dtype=pl.UInt16),
            "b": pl.Series([1, 3, 5], dtype=pl.Int16),
            "c": pl.Series([2, 4, 6], dtype=pl.Float32),
            "d": ["a", "b", "c"],
        }
    )
    q = df.unpivot(index="d")
    assert_gpu_result_equal(q)


def test_with_row_index_defaults():
    lf = pl.LazyFrame(
        {
            "a": [1, 3, 5],
            "b": [2, 4, 6],
        }
    )
    q = lf.with_row_index()
    assert_gpu_result_equal(q)


def test_unique_hash():
    # https://github.com/rapidsai/cudf/pull/19121#issuecomment-2959305678
    a = pl.LazyFrame({"a": [1, 2, 3]}).rename({"a": "A"})
    b = pl.LazyFrame({"a": [4, 5, 6]}).rename({"a": "A"})
    ir_a = Translator(a._ldf.visit(), pl.GPUEngine()).translate_ir()
    ir_b = Translator(b._ldf.visit(), pl.GPUEngine()).translate_ir()

    assert hash(ir_a) != hash(ir_b)
