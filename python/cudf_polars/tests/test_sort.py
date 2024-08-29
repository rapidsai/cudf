# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "sort_keys",
    [
        (pl.col("a"),),
        pytest.param(
            (pl.col("d").abs(),),
            marks=pytest.mark.xfail(reason="abs not yet implemented"),
        ),
        (pl.col("a"), pl.col("d")),
        (pl.col("b"),),
    ],
)
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("maintain_order", [False, True], ids=["unstable", "stable"])
def test_sort(sort_keys, nulls_last, maintain_order):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 1, 3, 5, None, None],
            "b": [1.5, 2.5, None, 1.5, 3, float("nan"), 3],
            "c": [True, True, True, True, False, False, True],
            "d": [1, 2, -1, 10, 6, -1, -7],
        }
    ).lazy()

    query = ldf.sort(
        *sort_keys,
        descending=True,
        nulls_last=nulls_last,
        maintain_order=maintain_order,
    )
    assert_gpu_result_equal(query, check_row_order=maintain_order)
