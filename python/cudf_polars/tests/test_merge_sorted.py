# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "descending",
    [
        pytest.param(
            True,
            marks=pytest.mark.xfail(reason="polars/issues/21511"),
        ),
        False,
    ],
)
def test_merge_sorted_without_nulls(descending):
    df0 = pl.LazyFrame(
        {"name": ["steve", "elise", "bob"], "age": [42, 44, 18], "height": [5, 6, 5]}
    ).sort("age", descending=descending)
    df1 = pl.LazyFrame(
        {
            "name": ["anna", "megan", "steve", "thomas"],
            "age": [21, 33, 42, 20],
            "height": [5, 5, 5, 5],
        }
    ).sort("age", descending=descending)
    q = df0.merge_sorted(df1, key="age")
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "descending",
    [
        pytest.param(
            True,
            marks=pytest.mark.xfail(reason="polars/issues/21511"),
        ),
        False,
    ],
)
def test_merge_sorted_with_nulls(descending):
    df0 = pl.LazyFrame(
        {
            "name": ["steve", "elise", "bob", "john"],
            "age": [42, 44, 18, None],
            "height": [5, 6, 7, 5],
        }
    ).sort("age", descending=descending)
    df1 = pl.LazyFrame(
        {
            "name": ["anna", "megan", "steve", "thomas", "john"],
            "age": [21, 33, 42, 20, None],
            "height": [5, 5, 5, 5, 5],
        }
    ).sort("age", descending=descending)
    q = df0.merge_sorted(df1, key="age")
    assert_gpu_result_equal(q)
