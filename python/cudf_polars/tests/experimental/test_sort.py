# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import DEFAULT_SCHEDULER, assert_gpu_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_130


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "scheduler": DEFAULT_SCHEDULER,
            "shuffle_method": "tasks",
            "fallback_mode": "raise",
        },
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7],
            "y": [1, 6, 7, 2, 5, 4, 3],
            "z": ["e", "c", "b", "g", "a", "f", "d"],
        }
    )


def test_sort(df, engine):
    q = df.sort(by=["y", "z"])
    if POLARS_VERSION_LT_130:
        with pytest.raises(
            pl.exceptions.ComputeError,
            match="Sort does not support multiple partitions.",
        ):
            assert_gpu_result_equal(q, engine=engine)
    else:
        with pytest.raises(
            NotImplementedError, match="Sort does not support multiple partitions."
        ):
            assert_gpu_result_equal(q, engine=engine)


def test_sort_head(df, engine):
    q = df.sort(by=["y", "z"]).head(2)
    assert_gpu_result_equal(q, engine=engine)


def test_sort_tail(df, engine):
    q = df.sort(by=["y", "z"]).tail(2)
    assert_gpu_result_equal(q, engine=engine)
