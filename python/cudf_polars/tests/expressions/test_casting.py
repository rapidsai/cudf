# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)

_supported_dtypes = [(pl.Int8(), pl.Int64())]

_unsupported_dtypes = [
    (pl.Datetime("ns"), pl.Int64()),
]


@pytest.fixture
def dtypes(request):
    return request.param


@pytest.fixture
def tests(dtypes):
    fromtype, totype = dtypes
    if fromtype == pl.String():
        data = ["a", "b", "c"]
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
