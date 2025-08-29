# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import DEFAULT_SCHEDULER, assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions


@pytest.mark.parametrize("rapidsmpf_spill", [False, True])
@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_join_rapidsmpf(
    max_rows_per_partition: int,
    rapidsmpf_spill: bool,  # noqa: FBT001
) -> None:
    # Check that we have a distributed cluster running.
    # This tests must be run with: --scheduler='distributed'
    distributed = pytest.importorskip("distributed")
    try:
        client = distributed.get_client()
    except ValueError:
        pytest.skip(reason="Requires distributed execution.")

    # check that we have a rapidsmpf cluster running
    pytest.importorskip("rapidsmpf")
    try:
        # This will result in a ValueError if the
        # scheduler isn't compatible with rapidsmpf.
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
            "scheduler": "distributed",
            "rapidsmpf_spill": rapidsmpf_spill,
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


@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_join_rapidsmpf_single(max_rows_per_partition: int) -> None:
    # check that we have a rapidsmpf cluster running
    pytest.importorskip("rapidsmpf")

    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "broadcast_join_limit": 2,
            "shuffle_method": "rapidsmpf",
            "scheduler": "synchronous",
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


def test_join_rapidsmpf_single_private_config() -> None:
    # The user may not specify "rapidsmpf-single" directly
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "shuffle_method": "rapidsmpf-single",
            "scheduler": "synchronous",
        },
    )
    with pytest.raises(ValueError, match="not a supported shuffle method"):
        ConfigOptions.from_polars_engine(engine)


def test_rapidsmpf_spill_synchronous_unsupported() -> None:
    # check that we have a rapidsmpf cluster running
    pytest.importorskip("rapidsmpf")

    # rapidsmpf_spill=True is not yet supported with synchronous scheduler.
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "shuffle_method": "rapidsmpf",
            "scheduler": "synchronous",
            "rapidsmpf_spill": True,
        },
    )
    with pytest.raises(ValueError, match="rapidsmpf_spill.*not supported.*synchronous"):
        ConfigOptions.from_polars_engine(engine)


@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_sort_rapidsmpf(max_rows_per_partition: int) -> None:
    # Require rapidsmpf, but don't require a distributed cluster,
    # because single-worker shuffle can be used.
    pytest.importorskip("rapidsmpf")

    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "shuffle_method": "rapidsmpf",
            "scheduler": DEFAULT_SCHEDULER,
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
