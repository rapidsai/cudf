# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_130

# TODO: Add multi-partition Sort support to the rapidsmpf runtime.
# See: https://github.com/rapidsai/cudf/issues/20486
pytestmark = pytest.mark.skipif(
    DEFAULT_RUNTIME == "rapidsmpf",
    reason="Sort not yet supported for rapidsmpf runtime.",
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
            "shuffle_method": "tasks",
            "fallback_mode": "raise",
        },
    )


@pytest.fixture(scope="module")
def engine_large():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 2_100,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
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


def large_frames():
    x = [1.0] * 10_000
    x[-1] = float("nan")
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 1000

    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
            }
        ),
        ["x"],
        False,
        id="all_equal_one_nan",
    )

    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
                "y": y,
            }
        ),
        ["x", "y"],
        False,
        id="two_cols",
    )

    idx = list(range(10_000))
    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
                "y": y,
                "idx": idx,
            }
        ),
        ["x", "y"],
        True,
        id="two_col_stable",
    )


def test_sort(df, engine):
    q = df.sort(by=["y", "z"])
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("large_df,by,stable", list(large_frames()))
@pytest.mark.parametrize(
    "nulls_last,descending", [(True, False), (True, True), (False, True)]
)
def test_large_sort(large_df, by, engine_large, stable, nulls_last, descending):
    q = large_df.sort(
        by=by, nulls_last=nulls_last, maintain_order=stable, descending=descending
    )
    assert_gpu_result_equal(q, engine=engine_large)


def test_sort_head(df, engine):
    q = df.sort(by=["y", "z"]).head(2)
    assert_gpu_result_equal(q, engine=engine)


def test_sort_tail(df, engine):
    q = df.sort(by=["y", "z"]).tail(2)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("offset", [1, -4])
def test_sort_slice(df, engine, offset):
    # Slice in the middle, which distributed sorts need to be careful with
    q = df.sort(by=["y", "z"]).slice(offset, 2)
    if POLARS_VERSION_LT_130:
        exception = pl.exceptions.ComputeError
    else:
        exception = NotImplementedError

    with pytest.raises(
        exception,
        match="Sort does not support a multi-partition slice with an offset.",
    ):
        assert_gpu_result_equal(q, engine=engine)


def test_sort_after_sparse_join():
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "max_rows_per_partition": 4,
        },
    )

    left = pl.LazyFrame({"foo": list(range(5)), "bar": list(range(5))})
    right = pl.LazyFrame({"foo": list(range(1))})
    q = left.join(right, on="foo", how="inner").sort(by=["foo"])
    assert_gpu_result_equal(q, engine=engine)
