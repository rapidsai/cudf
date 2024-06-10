# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("dtype", [pl.UInt32, pl.Int32, None])
def test_len(dtype):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    if dtype is None:
        q = df.select(pl.len())
    else:
        q = df.select(pl.len().cast(dtype))

    assert_gpu_result_equal(q)
