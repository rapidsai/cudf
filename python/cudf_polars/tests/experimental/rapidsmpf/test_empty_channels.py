# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for empty channel handling in RapidsMPF streaming nodes."""

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
def engine():
    """Standard GPU engine for tests."""
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 2,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )


def test_empty_dataframe_source(engine):
    """Test that an empty DataFrame source is handled correctly."""
    empty_df = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Int64, "b": pl.Int64})
    q = empty_df.lazy()
    assert_gpu_result_equal(q, engine=engine)


def test_join_with_empty_left(engine):
    """Test join where left side is empty."""
    left = pl.DataFrame({"key": [], "val": []}).cast({"key": pl.Int64, "val": pl.Int64})
    right = pl.DataFrame({"key": [1, 2], "other": [10, 20]})

    q = left.lazy().join(right.lazy(), on="key", how="inner")
    assert_gpu_result_equal(q, engine=engine)


def test_join_with_empty_right(engine):
    """Test join where right side is empty."""
    left = pl.DataFrame({"key": [1, 2], "val": [10, 20]})
    right = pl.DataFrame({"key": [], "other": []}).cast(
        {"key": pl.Int64, "other": pl.Int64}
    )

    q = left.lazy().join(right.lazy(), on="key", how="inner")
    assert_gpu_result_equal(q, engine=engine)


def test_join_with_both_empty(engine):
    """Test join where both sides are empty."""
    left = pl.DataFrame({"key": [], "val": []}).cast({"key": pl.Int64, "val": pl.Int64})
    right = pl.DataFrame({"key": [], "other": []}).cast(
        {"key": pl.Int64, "other": pl.Int64}
    )

    q = left.lazy().join(right.lazy(), on="key", how="inner")
    assert_gpu_result_equal(q, engine=engine)


def test_broadcast_join_empty_small_table(engine):
    """Test broadcast join with empty small (right) table."""
    # With max_rows_per_partition=2, 10 rows creates 5 partitions
    # This ensures broadcast_join_node is used instead of default_node_multi
    left = pl.DataFrame({"key": list(range(10)), "val": list(range(10, 20))})
    right = pl.DataFrame({"key": [], "other": []}).cast(
        {"key": pl.Int64, "other": pl.Int64}
    )

    q = left.lazy().join(right.lazy(), on="key", how="inner")
    assert_gpu_result_equal(q, engine=engine)
