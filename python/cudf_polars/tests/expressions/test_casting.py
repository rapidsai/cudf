# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_collect_raises,
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_130

_supported_dtypes = [(pl.Int8(), pl.Int64())]

_unsupported_dtypes = [
    (pl.Boolean(), pl.Datetime("ns")),
]


@pytest.fixture
def dtypes(request):
    return request.param


@pytest.fixture
def tests(dtypes):
    fromtype, totype = dtypes
    if fromtype == pl.String():
        data = ["a", "b", "c"]
    elif fromtype == pl.Boolean():
        data = [True, False, True]
    else:
        data = [1, 2, 3]
    return pl.DataFrame(
        {
            "a": pl.Series(data, dtype=fromtype),
        }
    ).lazy(), totype


@pytest.mark.parametrize("dtypes", _supported_dtypes, indirect=True)
def test_cast_supported(tests):
    df, totype = tests
    q = df.select(pl.col("a").cast(totype))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("dtypes", _unsupported_dtypes, indirect=True)
def test_cast_unsupported(tests):
    df, totype = tests
    assert_ir_translation_raises(
        df.select(pl.col("a").cast(totype)), NotImplementedError
    )


def test_allow_double_cast():
    df = pl.LazyFrame({"c0": [1000]})
    query = df.select(pl.col("c0").cast(pl.Boolean).cast(pl.Int8))
    assert_gpu_result_equal(query)


@pytest.mark.parametrize("dtype", [pl.Int64(), pl.Float64()])
@pytest.mark.parametrize("strict", [True, False])
def test_cast_strict_false_string_to_numeric(dtype, strict):
    df = pl.LazyFrame({"c0": ["1969-12-08 17:00:01", "1", None]})
    query = df.with_columns(pl.col("c0").cast(dtype, strict=strict))
    if strict:
        cudf_except = (
            pl.exceptions.ComputeError
            if POLARS_VERSION_LT_130
            else pl.exceptions.InvalidOperationError
        )
        assert_collect_raises(
            query,
            polars_except=pl.exceptions.InvalidOperationError,
            cudf_except=cudf_except,
        )
    else:
        assert_gpu_result_equal(query)


def test_cast_from_string_unsupported():
    df = pl.LazyFrame({"a": ["True"]})
    query = df.select(pl.col("a").cast(pl.Boolean()))
    assert_ir_translation_raises(query, NotImplementedError)


def test_cast_to_string_unsupported():
    df = pl.LazyFrame({"a": [True]})
    query = df.select(pl.col("a").cast(pl.String()))
    assert_ir_translation_raises(query, NotImplementedError)
