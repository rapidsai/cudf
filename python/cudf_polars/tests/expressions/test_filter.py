# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_filter():
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()

    # group-by is just to avoid the filter being pushed into the scan.
    query = ldf.group_by(pl.col("a")).agg(pl.col("b").sum()).filter(pl.col("b") < 1)
    assert_gpu_result_equal(query)
