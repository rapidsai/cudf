# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
)


@pytest.fixture
def decimal_df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": pl.Series(
                "a",
                [Decimal("0.10"), Decimal("1.10"), Decimal("100.10")],
                dtype=pl.Decimal(precision=9, scale=2),
            ),
        }
    )


def test_decimal_aggs(decimal_df: pl.LazyFrame) -> None:
    q = decimal_df.with_columns(
        sum=pl.col("a").sum(),
        min=pl.col("a").min(),
        max=pl.col("a").max(),
        mean=pl.col("a").mean(),
        median=pl.col("a").median(),
    )
    assert_gpu_result_equal(q)
