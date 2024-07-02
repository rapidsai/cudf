# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("maintain_order", [False, True], ids=["unstable", "stable"])
@pytest.mark.parametrize("pre_sorted", [False, True], ids=["unsorted", "sorted"])
def test_unique(maintain_order, pre_sorted):
    ldf = pl.DataFrame(
        {
            "b": [1.5, 2.5, None, 1.5, 3, float("nan"), 3],
        }
    ).lazy()
    if pre_sorted:
        ldf = ldf.sort("b")

    query = ldf.select(pl.col("b").unique(maintain_order=maintain_order))
    assert_gpu_result_equal(query, check_row_order=maintain_order)
