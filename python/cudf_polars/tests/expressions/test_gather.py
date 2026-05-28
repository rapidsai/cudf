# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_gather(engine: pl.GPUEngine):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, 3, 1, 5, 6, 1, 0],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.col("b")))
    assert_gpu_result_equal(query, engine=engine)


def test_gather_with_nulls(engine: pl.GPUEngine):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, None, 1, None, 6, 1, 0],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.col("b")))

    assert_gpu_result_equal(query, engine=engine)


def test_gather_empty_indices(engine: pl.GPUEngine):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.lit(pl.Series("idx", [], dtype=pl.Int64))))
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("negative", [False, True])
def test_gather_out_of_bounds(engine_raise_on_fail: pl.GPUEngine, negative):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, -10 if negative else 10, 1, 2, 6, 1, 0],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.col("b")))

    with pytest.raises(ValueError, match="gather indices are out of bounds"):
        query.collect(engine=engine_raise_on_fail)


@pytest.mark.parametrize(
    "idx",
    [
        0,
        pl.lit(0),
        pl.col("a").first(),
    ],
)
@pytest.mark.parametrize(
    "lit",
    [
        pl.lit(7),
        pytest.param(
            pl.lit([7]),
            marks=pytest.mark.xfail(
                reason="List literal loses nesting in gather: https://github.com/rapidsai/cudf/issues/19610"
            ),
        ),
        pl.lit([[7]]),
        pl.lit(pl.Series([7, 8, 9])),
    ],
)
def test_gather_on_literal(
    engine: pl.GPUEngine,
    lit: pl.Expr,
    idx: pl.Expr,
) -> None:
    df = pl.LazyFrame(
        {
            "g": [10, 10, 10, 20, 20, 30],
            "a": [0, 0, 0, 0, 0, 0],
            "b": [1, 1, 1, 1, 1, 1],
            "c": [11, 12, 13, 21, 22, 31],
        }
    )

    q = df.select(lit.gather(idx))
    assert_gpu_result_equal(q, engine=engine)
