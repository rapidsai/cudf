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

from cudf_polars.experimental.rapidsmpf.collectives.shuffle import (
    _is_already_partitioned,
)
from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)
from cudf_polars.utils.config import ConfigOptions, ShufflerInsertionMethod

REQUIRE_TASKS_RUNTIME = pytest.mark.skipif(
    DEFAULT_RUNTIME != "tasks", reason="Requires 'tasks' runtime."
)


@REQUIRE_TASKS_RUNTIME
@pytest.mark.parametrize("rapidsmpf_spill", [False, True])
@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
@pytest.mark.parametrize(
    "shuffler_insertion_method",
    [
        ShufflerInsertionMethod.INSERT_CHUNKS,
        ShufflerInsertionMethod.CONCAT_INSERT,
    ],
)
def test_join_rapidsmpf(
    max_rows_per_partition: int,
    *,
    rapidsmpf_spill: bool,
    shuffler_insertion_method: ShufflerInsertionMethod,
) -> None:
    # Check that we have a distributed cluster running.
    # This tests must be run with: --cluster='distributed'
    distributed = pytest.importorskip("distributed")
    try:
        client = distributed.get_client()
    except ValueError:
        pytest.skip(reason="Requires distributed execution.")

    # check that we have a rapidsmpf cluster running
    try:
        # This will result in a ValueError if the
        # cluster isn't compatible with rapidsmpf.
        # Otherwise, it's a no-op if the cluster
        # was already bootstrapped.
        from rapidsmpf.integrations.dask import bootstrap_dask_cluster

        bootstrap_dask_cluster(client)
    except ValueError:
        pytest.skip(reason="Requires a rapidsmpf-bootstrapped cluster.")

    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "broadcast_join_limit": 2,
            "shuffle_method": "rapidsmpf",
            "cluster": "distributed",
            "runtime": DEFAULT_RUNTIME,
            "rapidsmpf_spill": rapidsmpf_spill,
            "shuffler_insertion_method": shuffler_insertion_method,
        },
    )

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


@REQUIRE_TASKS_RUNTIME
@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_join_rapidsmpf_single(max_rows_per_partition: int) -> None:
    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "broadcast_join_limit": 2,
            "shuffle_method": "rapidsmpf",
            "cluster": "single",
            "runtime": DEFAULT_RUNTIME,
        },
    )

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


def test_rapidsmpf_spill_single_unsupported() -> None:
    # rapidsmpf_spill=True is not yet supported with single-GPU cluster.
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "shuffle_method": "rapidsmpf",
            "cluster": "single",
            "runtime": DEFAULT_RUNTIME,
            "rapidsmpf_spill": True,
        },
    )
    with pytest.raises(ValueError, match="rapidsmpf_spill.*not supported.*single"):
        ConfigOptions.from_polars_engine(engine)


@REQUIRE_TASKS_RUNTIME
@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_sort_rapidsmpf(max_rows_per_partition: int) -> None:
    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "shuffle_method": "rapidsmpf",
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )

    df = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    q = df.sort(by=["y", "z"])

    assert_gpu_result_equal(q, engine=engine, check_row_order=True)


@REQUIRE_TASKS_RUNTIME
def test_sort_stable_rapidsmpf_warns():
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "shuffle_method": "rapidsmpf",
            "fallback_mode": "warn",
        },
    )

    df = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )

    q = df.sort(by=["y", "z"], maintain_order=True)
    with pytest.warns(UserWarning, match="Falling back to shuffle_method='tasks'."):
        assert_gpu_result_equal(q, engine=engine, check_row_order=True)


def test_is_already_partitioned():
    # Unit test for _is_already_partitioned helper
    chunks = 4
    columns = (0, 1)
    modulus = 8

    # Exact match: should return True
    metadata_match = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_match, columns, modulus) is True

    # Different columns: should return False
    metadata_diff_cols = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme((0,), modulus),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_diff_cols, columns, modulus) is False

    # Different local partitioning: should return False
    metadata_diff_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=None,
        ),
    )
    assert _is_already_partitioned(metadata_diff_local, columns, modulus) is False

    # Different modulus: should return False
    metadata_diff_mod = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, 16),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_diff_mod, columns, modulus) is False

    # No partitioning: should return False
    metadata_none = ChannelMetadata(chunks)
    assert _is_already_partitioned(metadata_none, columns, modulus) is False

    # Local not "inherit": should return False
    metadata_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=HashScheme((0,), 4),
        ),
    )
    assert _is_already_partitioned(metadata_local, columns, modulus) is False
