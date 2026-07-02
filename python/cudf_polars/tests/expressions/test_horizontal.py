# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)

if TYPE_CHECKING:
    from collections.abc import Callable


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
@pytest.mark.parametrize(
    "fn", [pl.sum_horizontal, pl.mean_horizontal], ids=["sum", "mean"]
)
def test_sum_mean_horizontal(
    df: pl.LazyFrame,
    engine: pl.GPUEngine,
    fn: Callable[..., pl.Expr],
    ignore_nulls: bool,  # noqa: FBT001
) -> None:
    q = df.select(fn("a", "b", "c", ignore_nulls=ignore_nulls))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize(
    "fn", [pl.min_horizontal, pl.max_horizontal], ids=["min", "max"]
)
def test_min_max_horizontal(
    df: pl.LazyFrame, engine: pl.GPUEngine, fn: Callable[..., pl.Expr]
) -> None:
    q = df.select(fn("a", "b", "c"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "fn",
    [pl.sum_horizontal, pl.min_horizontal, pl.max_horizontal, pl.mean_horizontal],
    ids=["sum", "min", "max", "mean"],
)
def test_horizontal_single_column(
    df: pl.LazyFrame, engine: pl.GPUEngine, fn: Callable[..., pl.Expr]
) -> None:
    q = df.select(fn("a"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize(
    "fn", [pl.sum_horizontal, pl.mean_horizontal], ids=["sum", "mean"]
)
def test_sum_mean_horizontal_mixed_dtypes(
    engine: pl.GPUEngine, fn: Callable[..., pl.Expr]
) -> None:
    df = pl.LazyFrame({"a": [1, 2, None], "b": [1.5, None, 3.5]})
    q = df.select(fn("a", "b"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize(
    "fn", [pl.min_horizontal, pl.max_horizontal], ids=["min", "max"]
)
def test_min_max_horizontal_all_null_row(
    engine: pl.GPUEngine, fn: Callable[..., pl.Expr]
) -> None:
    df = pl.LazyFrame(
        {"a": [None, 1], "b": [None, 2]}, schema={"a": pl.Int64, "b": pl.Int64}
    )
    q = df.select(fn("a", "b"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "fn", [pl.min_horizontal, pl.max_horizontal], ids=["min", "max"]
)
def test_min_max_horizontal_string_unsupported(
    engine: pl.GPUEngine, fn: Callable[..., pl.Expr]
) -> None:
    df = pl.LazyFrame({"a": ["x", None], "b": ["y", "z"]})
    q = df.select(fn("a", "b"))
    assert_ir_translation_raises(q, engine, NotImplementedError)
