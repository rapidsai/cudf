# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        "semi",
        "anti",
        "full",
    ],
)
@pytest.mark.parametrize("coalesce", [False, True])
@pytest.mark.parametrize(
    "join_nulls", [False, True], ids=["nulls_not_equal", "nulls_equal"]
)
@pytest.mark.parametrize(
    "join_expr",
    [
        pl.col("a"),
        pl.col("a") * 2,
        [pl.col("a"), pl.col("c") + 1],
        ["c", "a"],
    ],
)
def test_join(how, coalesce, join_nulls, join_expr):
    left = pl.DataFrame(
        {
            "a": [1, 2, 3, 1, None],
            "b": [1, 2, 3, 4, 5],
            "c": [2, 3, 4, 5, 6],
        }
    ).lazy()
    right = pl.DataFrame(
        {
            "a": [1, 4, 3, 7, None, None],
            "c": [2, 3, 4, 5, 6, 7],
        }
    ).lazy()

    query = left.join(
        right, on=join_expr, how=how, join_nulls=join_nulls, coalesce=coalesce
    )
    assert_gpu_result_equal(query, check_row_order=how == "left")


def test_cross_join():
    left = pl.DataFrame(
        {
            "a": [1, 2, 3, 1, None],
            "b": [1, 2, 3, 4, 5],
            "c": [2, 3, 4, 5, 6],
        }
    ).lazy()
    right = pl.DataFrame(
        {
            "a": [1, 4, 3, 7, None, None],
            "c": [2, 3, 4, 5, 6, 7],
        }
    ).lazy()

    q = left.join(right, how="cross")

    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "left_on,right_on", [(pl.col("a"), pl.lit(2)), (pl.lit(2), pl.col("a"))]
)
def test_join_literal_key_unsupported(left_on, right_on):
    left = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    right = pl.LazyFrame({"a": [1, 2, 3], "b": [5, 6, 7]})
    q = left.join(right, left_on=left_on, right_on=right_on, how="inner")

    assert_ir_translation_raises(q, NotImplementedError)
