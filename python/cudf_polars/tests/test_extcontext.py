# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_extcontext():
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()
    ldf2 = ldf.select((pl.col("b") + pl.col("a")).alias("c"))
    query = ldf.with_context(ldf2).select(pl.col("b"), pl.col("c"))
    with pytest.raises(pl.exceptions.ComputeError):
        # ExtContext to be deprecated so we're not implementing it.
        assert_gpu_result_equal(query)
