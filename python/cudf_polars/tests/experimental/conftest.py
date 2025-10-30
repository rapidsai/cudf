# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest


# scope="session" is important to not cause singificant slowdowns in CI
# https://github.com/rapidsai/cudf/pull/20137
@pytest.fixture(autouse=True, scope="session")
def dask_cluster(pytestconfig, worker_id):
    if (
        pytestconfig.getoption("--cluster") == "distributed"
        and pytestconfig.getoption("--executor") == "streaming"
    ):
        worker_count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0"))
        from dask import config
        from dask_cuda import LocalCUDACluster

        # Avoid "Sending large graph of size ..." warnings
        # (We expect these for tests using literal/random arrays)
        config.set({"distributed.admin.large-graph-warning-threshold": "20MB"})
        if worker_count > 0:
            # Avoid port conflicts with multiple test runners
            worker_index = int(worker_id.removeprefix("gw"))
            scheduler_port = 8800 + worker_index
            dashboard_address = 8900 + worker_index
        else:
            scheduler_port = None
            dashboard_address = None

        n_workers = int(os.environ.get("CUDF_POLARS_NUM_WORKERS", "1"))

        with (
            LocalCUDACluster(
                n_workers=n_workers,
                scheduler_port=scheduler_port,
                dashboard_address=dashboard_address,
            ) as cluster,
            cluster.get_client(),
        ):
            yield
    else:
        yield
