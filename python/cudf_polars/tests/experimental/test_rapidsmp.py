# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_join_rapidsmp(max_rows_per_partition: int) -> None:
    # Check that we have a distributed cluster running.
    # This tests must be run with:
    # --executor='dask-experimental' --dask-cluster --rapidsmp
    distributed = pytest.importorskip("distributed")
    try:
        client = distributed.get_client()
    except ValueError:
        pytest.skip(reason="Requires distributed execution.")

    # check that we have a rapidsmp cluster running
    rapidsmp = pytest.importorskip("rapidsmp")
    try:
        # This will result in a ValueError if the
        # scheduler isn't compatible with rapidsmp.
        # Otherwise, it's a no-op if the cluster
        # was already bootstrapped.
        rapidsmp.integrations.dask.bootstrap_dask_cluster(client)
    except ValueError:
        pytest.skip(reason="Requires rapidsmp cluster.")

    # Setup the GPUEngine config
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "broadcast_join_limit": 2,
            "shuffle_method": "rapidsmp",
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
