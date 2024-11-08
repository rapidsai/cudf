# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("then_scalar", [False, True])
@pytest.mark.parametrize("otherwise_scalar", [False, True])
@pytest.mark.parametrize("expr", [pl.col("c"), pl.col("c").is_not_null()])
def test_when_then(then_scalar, otherwise_scalar, expr):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [10, 13, 11, 15, 16, 11, 10],
            "c": [None, True, False, False, True, True, False],
        }
    )

    then = pl.lit(10) if then_scalar else pl.col("a")
    otherwise = pl.lit(-2) if otherwise_scalar else pl.col("b")
    q = ldf.select(pl.when(expr).then(then).otherwise(otherwise))
    assert_gpu_result_equal(q)
