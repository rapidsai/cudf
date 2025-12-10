# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for empty channel handling in RapidsMPF streaming nodes."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
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
    """Test that an empty DataFrame source produces 0 chunks."""
    empty_df = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Int64, "b": pl.Int64})
    result = empty_df.lazy().collect(engine=engine)
    assert result.height == 0
    assert result.columns == ["a", "b"]


def test_filter_on_empty_dataframe(engine):
    """Test filter on empty DataFrame."""
    empty_df = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Int64, "b": pl.Int64})
    result = empty_df.lazy().filter(pl.col("a") > 0).collect(engine=engine)
    assert result.height == 0


def test_join_with_empty_left(engine):
    """Test join where left side is empty."""
    left = pl.DataFrame({"key": [], "val": []}).cast({"key": pl.Int64, "val": pl.Int64})
    right = pl.DataFrame({"key": [1, 2], "other": [10, 20]})

    result = (
        left.lazy().join(right.lazy(), on="key", how="inner").collect(engine=engine)
    )
    assert result.height == 0


def test_join_with_empty_right(engine):
    """Test join where right side is empty."""
    left = pl.DataFrame({"key": [1, 2], "val": [10, 20]})
    right = pl.DataFrame({"key": [], "other": []}).cast(
        {"key": pl.Int64, "other": pl.Int64}
    )

    result = (
        left.lazy().join(right.lazy(), on="key", how="inner").collect(engine=engine)
    )
    assert result.height == 0


def test_join_with_both_empty(engine):
    """Test join where both sides are empty."""
    left = pl.DataFrame({"key": [], "val": []}).cast({"key": pl.Int64, "val": pl.Int64})
    right = pl.DataFrame({"key": [], "other": []}).cast(
        {"key": pl.Int64, "other": pl.Int64}
    )

    result = (
        left.lazy().join(right.lazy(), on="key", how="inner").collect(engine=engine)
    )
    assert result.height == 0


def test_broadcast_join_empty_small_table(engine):
    """Test broadcast join with empty small (right) table."""
    # With max_rows_per_partition=2, 10 rows creates 5 partitions
    left = pl.DataFrame({"key": list(range(10)), "val": list(range(10, 20))})
    right = pl.DataFrame({"key": [], "other": []}).cast(
        {"key": pl.Int64, "other": pl.Int64}
    )

    result = (
        left.lazy().join(right.lazy(), on="key", how="inner").collect(engine=engine)
    )
    assert result.height == 0
