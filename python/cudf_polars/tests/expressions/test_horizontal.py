# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": [1, None, None, 4, 5],
            "b": [10, 20, None, None, 50],
            "c": [100, 200, None, 400, None],
        }
    )


@pytest.mark.parametrize("ignore_nulls", [True, False])
def test_mean_horizontal(
    df: pl.LazyFrame,
    engine: pl.GPUEngine,
    ignore_nulls: bool,  # noqa: FBT001
) -> None:
    q = df.select(pl.mean_horizontal("a", "b", "c", ignore_nulls=ignore_nulls))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_mean_horizontal_single_column(df: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    q = df.select(pl.mean_horizontal("a"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_mean_horizontal_mixed_dtypes(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, None], "b": [1.5, None, 3.5]})
    q = df.select(pl.mean_horizontal("a", "b"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)
