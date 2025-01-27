# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "offset",
    [0, 1, 2, -10, -20, -1, -2, 20],
)
@pytest.mark.parametrize(
    "length",
    [0, 2, 12, 11],
)
@pytest.mark.parametrize("slice_pushdown", [False, True])
def test_slice(offset, length, slice_pushdown):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()

    query = (
        ldf.group_by(pl.col("a"))
        .agg(pl.col("b").sum())
        .sort(by=pl.col("a"))
        .slice(offset, length)
    )
    assert_gpu_result_equal(query, collect_kwargs={"slice_pushdown": slice_pushdown})
