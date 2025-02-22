# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_slice():
    df = pl.LazyFrame({"a": [0, 1, 2, 3], "b": [1, 2, 3, 4]})
    q = df.select(pl.col("a").slice(1))

    assert_gpu_result_equal(q)

    q = df.select(pl.col("a").slice(1, 3))

    assert_gpu_result_equal(q)
