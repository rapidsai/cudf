# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("subset", [None, ["a"], ["a", "b"], ["b", "c"]])
@pytest.mark.parametrize("keep", ["any", "none", "first", "last"])
@pytest.mark.parametrize("maintain_order", [False, True], ids=["unstable", "stable"])
def test_distinct(subset, keep, maintain_order):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 1, 3, 5, None, None],
            "b": [1.5, 2.5, None, 1.5, 3, float("nan"), 3],
            "c": [True, True, True, True, False, False, True],
        }
    ).lazy()

    query = ldf.unique(subset=subset, keep=keep, maintain_order=maintain_order)
    assert_gpu_result_equal(query, check_row_order=maintain_order)
