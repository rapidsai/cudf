# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "when",
    [
        pl.lit(value=True, dtype=pl.Boolean()),
        pl.lit(value=False, dtype=pl.Boolean()),
        pl.lit(None, dtype=pl.Boolean()),
        pl.col("c"),
    ],
)
@pytest.mark.parametrize("then", [pl.lit(10), pl.col("a")])
@pytest.mark.parametrize("otherwise", [pl.lit(-2), pl.col("b")])
def test_when_then(when, then, otherwise):
    df = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [10, 13, 11, 15, 16, 11, 10],
            "c": [None, True, False, False, True, True, False],
        }
    )

    q = df.select(pl.when(when).then(then).otherwise(otherwise))
    assert_gpu_result_equal(q)
