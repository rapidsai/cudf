# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamic join path in join_actor (including Right and Full joins)."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)

pytestmark = [
    pytest.mark.skipif(
        DEFAULT_RUNTIME != "rapidsmpf", reason="Requires 'rapidsmpf' runtime."
    ),
    pytest.mark.skipif(
        DEFAULT_CLUSTER != "single", reason="Requires 'single' cluster."
    ),
]


@pytest.fixture
def left():
    return pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )


@pytest.fixture
def right():
    return pl.LazyFrame(
        {
            "xx": range(9),
            "y": [2, 4, 3] * 3,
            "zz": [1, 2, 3] * 3,
        }
    )


@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize("max_rows_per_partition", [3, 5])
def test_dynamic_join_how(left, right, how, max_rows_per_partition):
    """Dynamic join path: all join types including Right and Full."""
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "broadcast_join_limit": 2,
            "shuffle_method": "rapidsmpf",
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "dynamic_planning": {},
        },
    )
    q = left.join(right, on="y", how=how)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("how", ["right", "full"])
def test_dynamic_join_right_full_reverse(left, right, how):
    """Dynamic join path: Right/Full with reversed left/right (stress ordering)."""
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "broadcast_join_limit": 2,
            "shuffle_method": "rapidsmpf",
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "dynamic_planning": {},
        },
    )
    # Reverse so "right" frame is larger; exercises right-side preservation
    q = right.join(left, on="y", how=how)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
