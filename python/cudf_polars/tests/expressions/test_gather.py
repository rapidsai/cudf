# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_gather():
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, 3, 1, 5, 6, 1, 0],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.col("b")))
    assert_gpu_result_equal(query)
