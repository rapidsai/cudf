# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("dtype", [pl.UInt32, pl.Int32, None])
@pytest.mark.parametrize("empty", [False, True])
def test_len(dtype, empty):
    if empty:
        df = pl.LazyFrame({})
    else:
        df = pl.LazyFrame({"a": [1, 2, 3]})

    if dtype is None:
        q = df.select(pl.len())
    else:
        q = df.select(pl.len().cast(dtype))

    # Workaround for https://github.com/pola-rs/polars/issues/16904
    assert_gpu_result_equal(
        q,
        collect_kwargs={"optimizations": pl.QueryOptFlags(projection_pushdown=False)},
    )


@pytest.mark.parametrize("data", [[1, 2, 3], [1, 2, None]])
def test_col_len(data):
    data = {"a": list("xyz"), "b": data}
    q = pl.LazyFrame(data).select(
        pl.col("a").len().alias("l"),
        (pl.col("a").len() * 2).alias("l2"),
        pl.col("b").len().alias("l3"),
    )
    assert_gpu_result_equal(q)
