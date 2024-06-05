# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_select_series_literal():
    df = pl.LazyFrame({})

    q = df.select(
        a=pl.Series([1, 2, 3]),
        b=pl.Series(["a", "b", "c"], dtype=pl.String()),
        c=pl.Series([[1, 2], [3], None], dtype=pl.List(pl.UInt16())),
    )

    assert_gpu_result_equal(q)
