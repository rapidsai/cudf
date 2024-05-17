# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        "outer",
        "semi",
        "anti",
        pytest.param(
            "cross",
            marks=pytest.mark.xfail(reason="cross join not implemented"),
        ),
        "outer_coalesce",
    ],
)
@pytest.mark.parametrize(
    "join_nulls", [False, True], ids=["nulls_not_equal", "nulls_equal"]
)
@pytest.mark.parametrize(
    "join_expr",
    [
        pl.col("a"),
        pl.col("a") * 2,
        [pl.col("a"), pl.col("a") + 1],
        ["c", "a"],
    ],
)
def test_join(request, how, join_nulls, join_expr):
    request.applymarker(
        pytest.mark.xfail(
            how == "outer_coalesce"
            and isinstance(join_expr, list)
            and not isinstance(join_expr[0], str),
            reason="https://github.com/pola-rs/polars/issues/16289",
        )
    )
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

    query = left.join(right, on=join_expr, how=how, join_nulls=join_nulls)
    assert_gpu_result_equal(query, check_row_order=False)
