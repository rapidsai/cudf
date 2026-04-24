# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_135, POLARS_VERSION_LT_136


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


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("with_nulls", ["no_nulls", "nulls"])
def test_setsorted(request, descending, nulls_last, with_nulls):
    request.applymarker(
        pytest.mark.xfail(
            condition=not POLARS_VERSION_LT_135 and POLARS_VERSION_LT_136,
            reason="HintIR not supported",
        )
    )
    values = sorted([1, 2, 3, 4, 5, 6, -2], reverse=descending)
    if with_nulls == "nulls":
        values[-1 if nulls_last else 0] = None
    df = pl.LazyFrame({"a": values})

    q = df.set_sorted("a", descending=descending)

    assert_gpu_result_equal(q)


def test_sort_concat_filtered_to_empty():
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = pl.concat([df.filter(pl.col("a") == 0), df.filter(pl.col("a") == 4)]).sort("a")
    assert_gpu_result_equal(q)
