# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)

import polars as pl

from cudf_polars.experimental.rapidsmpf.utils import _is_already_partitioned
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {
                "max_rows_per_partition": 1,
                "broadcast_join_limit": 2,
                "shuffle_method": "rapidsmpf",
            }
        },
        {
            "executor_options": {
                "max_rows_per_partition": 5,
                "broadcast_join_limit": 2,
                "shuffle_method": "rapidsmpf",
            }
        },
    ],
    indirect=True,
)
def test_join_rapidsmpf(engine) -> None:
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(6),
            "y": [2, 4, 3] * 2,
            "zz": [1, 2] * 3,
        }
    )
    q = left.join(right, on="y", how="inner")
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {
                "max_rows_per_partition": 1,
                "shuffle_method": "rapidsmpf",
            }
        },
        {
            "executor_options": {
                "max_rows_per_partition": 5,
                "shuffle_method": "rapidsmpf",
            }
        },
    ],
    indirect=True,
)
def test_sort_rapidsmpf(engine) -> None:
    df = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    q = df.sort(by=["y", "z"])
    assert_gpu_result_equal(q, engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "join_expr",
    [
        pl.col("y") * 2,
        [pl.col("y"), pl.col("a") + 1],
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {
                "max_rows_per_partition": 1,
                "broadcast_join_limit": 2,
                "shuffle_method": "rapidsmpf",
            }
        },
    ],
    indirect=True,
)
def test_join_non_col_keys_rapidsmpf(engine, join_expr) -> None:
    """Non-Col (expression) shuffle keys should work with the rapidsmpf backend."""
    left = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4] * 3,
            "y": [10, 20, 30, 40] * 3,
            "val_l": range(12),
        }
    )
    right = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4] * 2,
            "y": [10, 20, 30, 40] * 2,
            "val_r": range(8),
        }
    )
    q = left.join(right, on=join_expr, how="inner", coalesce=True)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_is_already_partitioned():
    # Unit test for _is_already_partitioned helper
    chunks = 4
    columns = (0, 1)
    modulus = 8
    nranks = 1

    # Exact match: should return True
    metadata_match = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_match, columns, modulus, nranks) is True

    # Different columns: should return False
    metadata_diff_cols = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme((0,), modulus),
            local="inherit",
        ),
    )
    assert (
        _is_already_partitioned(metadata_diff_cols, columns, modulus, nranks) is False
    )

    # Different local partitioning: should return False
    metadata_diff_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=None,
        ),
    )
    assert (
        _is_already_partitioned(metadata_diff_local, columns, modulus, nranks) is False
    )

    # Different modulus: should return False
    metadata_diff_mod = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, 16),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_diff_mod, columns, modulus, nranks) is False

    # No partitioning: should return False
    metadata_none = ChannelMetadata(chunks)
    assert _is_already_partitioned(metadata_none, columns, modulus, nranks) is False

    # Local not "inherit": should return False
    metadata_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=HashScheme((0,), 4),
        ),
    )
    assert _is_already_partitioned(metadata_local, columns, modulus, nranks) is False
