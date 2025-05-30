# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_struct_field():
    ldf = pl.LazyFrame({"a": [{"b": "c", "d": "e"}]})
    query = ldf.select(pl.col("a").struct.field("b"))
    assert_gpu_result_equal(query)
