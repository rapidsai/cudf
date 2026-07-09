# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_replace_strict(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 2, None, 3]})
    q = df.select(
        pl.col("a").replace_strict([1, 2], [10, 20], default=-1).alias("list_replace"),
        pl.col("a").replace_strict([1, 2], 99, default=-1).alias("broadcast_replace"),
    )
    assert_gpu_result_equal(q, engine=engine)
