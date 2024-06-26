# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_sort_expression(descending, nulls_last):
    ldf = pl.LazyFrame(
        {
            "a": [5, -1, 3, 4, None, 8, 6, 7, None],
        }
    )

    query = ldf.select(pl.col("a").sort(descending=descending, nulls_last=nulls_last))
    assert_gpu_result_equal(query)


@pytest.mark.parametrize(
    "descending", itertools.combinations_with_replacement([False, True], 3)
)
@pytest.mark.parametrize(
    "nulls_last", itertools.combinations_with_replacement([False, True], 3)
)
@pytest.mark.parametrize("maintain_order", [False, True], ids=["unstable", "stable"])
def test_sort_by_expression(descending, nulls_last, maintain_order):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [1, 2, 2, 3, 9, 5, -1, 2, -2, 16],
            "c": ["a", "A", "b", "b", "c", "d", "A", "Z", "ä", "̈Ä"],
        }
    )

    query = ldf.select(
        pl.col("a").sort_by(
            pl.col("b"),
            pl.col("c"),
            pl.col("b") + pl.col("a"),
            descending=descending,
            nulls_last=nulls_last,
            maintain_order=maintain_order,
        )
    )
    assert_gpu_result_equal(query, check_row_order=maintain_order)
