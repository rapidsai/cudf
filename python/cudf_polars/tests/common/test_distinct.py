# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("subset", [None, ["a"], ["a", "b"], ["b", "c"], ["c", "a"]])
@pytest.mark.parametrize("keep", ["any", "none", "first", "last"])
@pytest.mark.parametrize("maintain_order", [False, True], ids=["unstable", "stable"])
@pytest.mark.parametrize("pre_sorted", [False, True], ids=["unsorted", "sorted"])
def test_distinct(subset, keep, maintain_order, pre_sorted):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 1, 3, 5, None, None],
            "b": [1.5, 2.5, None, 1.5, 3, float("nan"), 3],
            "c": [True, True, True, True, False, False, True],
        }
    ).lazy()
    if pre_sorted:
        keys = ["a", "b", "c"] if subset is None else subset
        descending = False if len(keys) == 1 else [False, True, True][: len(keys)]
        ldf = ldf.sort(*keys, descending=descending)

    query = ldf.unique(subset=subset, keep=keep, maintain_order=maintain_order)
    assert_gpu_result_equal(query, check_row_order=maintain_order)
