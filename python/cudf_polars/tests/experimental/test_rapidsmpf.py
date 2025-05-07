# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("rapidsmpf_spill", [False, True])
@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_join_rapidsmpf(
    max_rows_per_partition: int,
    rapidsmpf_spill: bool,  # noqa: FBT001
) -> None:
    # Check that we have a distributed cluster running.
    # This tests must be run with:
    # --executor='streaming' --scheduler='distributed'
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
