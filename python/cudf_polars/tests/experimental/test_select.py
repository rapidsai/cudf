# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
            "c": [2, 4, 6, 8, 10, 12, 14],
        }
    )


def test_select(df, engine):
    query = df.select(
        pl.col("a") + pl.col("b"), (pl.col("a") * 2 + pl.col("b")).alias("d")
    )
    assert_gpu_result_equal(query, engine=engine)


def test_select_unsupported_raises(df, engine):
    query = df.select(
        (pl.col("a") + pl.col("b")).max(),
        # NOTE: We don't support `median` yet
        (pl.col("a") * 2 + pl.col("b")).alias("d").median(),
    )
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="NotImplementedError",
    ):
        assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize(
    "aggs",
    [
        (
            (pl.col("a") + pl.col("b")).sum(),
            (pl.col("a") * 2 + pl.col("b")).alias("d").min(),
        ),
        (pl.col("a").min() + pl.col("b").max(),),
        (pl.col("a") - (pl.col("b") + pl.col("c").max()).sum(),),
        (pl.col("b").len(),),
        (pl.col("a") - (pl.col("b") + pl.col("c").max()).mean(),),
        (pl.col("b").n_unique(), pl.col("c").n_unique()),
    ],
)
def test_select_aggs(df, engine, aggs):
    # Test supported aggs (e.g. "min", "max", "std")
    query = df.select(*aggs)
    assert_gpu_result_equal(query, engine=engine)


def test_select_with_cse_no_agg(df, engine):
    expr = pl.col("a") + pl.col("a")
    query = df.select(expr, (expr * 2).alias("b"), ((expr * 2) + 10).alias("c"))
    assert_gpu_result_equal(query, engine=engine)
