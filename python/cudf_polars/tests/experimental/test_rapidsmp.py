# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal

try:
    from distributed import Client
    from distributed.utils_test import (  # noqa: F401
        cleanup,
        gen_test,
        loop,
        loop_in_thread,
    )
    from rapidsmp.integrations.dask import (
        LocalRMPCluster,
        bootstrap_dask_cluster,
    )
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize("max_rows_per_partition", [1, 5])
def test_join_rapidsmp(
    loop: pytest.FixtureDef,
    max_rows_per_partition: int,
) -> None:
    with LocalRMPCluster(loop=loop) as cluster:  # noqa: SIM117
        with Client(cluster) as client:
            bootstrap_dask_cluster(client)

            engine = pl.GPUEngine(
                raise_on_fail=True,
                executor="dask-experimental",
                executor_options={
                    "max_rows_per_partition": max_rows_per_partition,
                    "bcast_join_limit": 2,
                    "shuffle_method": "rapidsmp",
                },
            )
            left = pl.LazyFrame(
                {
                    "x": range(15),
                    "y": ["cat", "dog", "fish"] * 5,
                    "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
                }
            )
            right = pl.LazyFrame(
                {
                    "xx": range(6),
                    "y": ["dog", "bird", "fish"] * 2,
                    "zz": [1, 2] * 3,
                }
            )

            q = left.join(right, on="y", how="inner")

            assert_gpu_result_equal(q, engine=engine, check_row_order=False)
