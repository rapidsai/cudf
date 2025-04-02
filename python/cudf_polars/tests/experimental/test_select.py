# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        executor_options={"max_rows_per_partition": 3},
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    )


def test_select(df, engine):
    query = df.select(
        pl.col("a") + pl.col("b"), (pl.col("a") * 2 + pl.col("b")).alias("d")
    )
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("fallback_mode", ["silent", "raise", "warn"])
def test_select_reduce_fallback(df, fallback_mode):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        executor_options={
            "fallback_mode": fallback_mode,
            "max_rows_per_partition": 3,
        },
    )
    match = "does not support multiple partitions"

    query = df.select(
        (pl.col("a") + pl.col("b")).max(),
        (pl.col("a") * 2 + pl.col("b")).alias("d").mean(),
    )

    if fallback_mode == "silent":
        ctx = contextlib.nullcontext()
    elif fallback_mode == "raise":
        ctx = pytest.raises(pl.exceptions.ComputeError, match=match)
    else:
        ctx = pytest.warns(UserWarning, match=match)
    with ctx:
        assert_gpu_result_equal(query, engine=engine)


def test_select_with_cse_no_agg(df, engine):
    expr = pl.col("a") + pl.col("a")
    query = df.select(expr, (expr * 2).alias("b"), ((expr * 2) + 10).alias("c"))
    assert_gpu_result_equal(query, engine=engine)
