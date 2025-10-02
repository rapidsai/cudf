# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def df():
    return pl.LazyFrame(
        {
            "test": [2, 4, 1, 3],
            "val": [2, 4, 9, 3],
            "bool_val": [False, True, True, False],
            "str_value": ["d", "b", "a", "c"],
            "col_with_nulls": [2, 4, None, 3],
        }
    )


@pytest.mark.parametrize("col", ["test", "bool_val", "str_value", "col_with_nulls"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_top_k(df, col, k):
    q = df.select(pl.col(col).top_k(k))
    assert_gpu_result_equal(q, check_row_order=False)


@pytest.mark.parametrize("col", ["test", "bool_val", "str_value", "col_with_nulls"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_bottom_k(df, col, k):
    q = df.select(pl.col(col).bottom_k(k))
    assert_gpu_result_equal(q, check_row_order=False)
