# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
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


def test_filter_pointwise(df, engine):
    query = df.filter(pl.col("a") > 3)
    assert_gpu_result_equal(query, engine=engine)


def test_filter_non_pointwise(df, engine):
    query = df.filter(pl.col("a") > pl.col("a").max())
    with pytest.warns(
        UserWarning, match="This filter is not supported for multiple partitions."
    ):
        assert_gpu_result_equal(query, engine=engine)


def test_filter_selectivity_hint(df):
    # Test that selectivity hints trigger repartition after selective filters
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            # Hint to reduce partitions after this filter
            "selectivity_hints": {"FILTER ('a', 'b', 'c')": 0.1},
        },
    )
    query = df.filter(pl.col("a") > 3)
    assert_gpu_result_equal(query, engine=engine)
