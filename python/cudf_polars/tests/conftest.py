# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest


@pytest.fixture(params=[False, True], ids=["no_nulls", "nulls"], scope="session")
def with_nulls(request):
    return request.param


def pytest_addoption(parser):
    parser.addoption(
        "--executor",
        action="store",
        default="streaming",
        choices=("in-memory", "streaming"),
        help="Executor to use for GPUEngine.",
    )

    parser.addoption(
        "--scheduler",
        action="store",
        default="synchronous",
        choices=("synchronous", "distributed"),
        help="Scheduler to use for 'streaming' executor.",
    )

    parser.addoption(
        "--blocksize-mode",
        action="store",
        default="default",
        choices=("small", "default"),
        help=(
            "Blocksize to use for 'streaming' executor. Set to 'small' "
            "to run most tests with multiple partitions."
        ),
    )


@pytest.fixture(scope="session", autouse=True)
def dask_cluster(pytestconfig, worker_id):
    if (
        pytestconfig.getoption("--scheduler") == "distributed"
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


def pytest_configure(config):
    import cudf_polars.testing.asserts

    if (
        config.getoption("--scheduler") == "distributed"
        and config.getoption("--executor") != "streaming"
    ):
        raise pytest.UsageError("Distributed scheduler requires --executor='streaming'")

    cudf_polars.testing.asserts.DEFAULT_EXECUTOR = config.getoption("--executor")
    cudf_polars.testing.asserts.DEFAULT_SCHEDULER = config.getoption("--scheduler")
    cudf_polars.testing.asserts.DEFAULT_BLOCKSIZE_MODE = config.getoption(
        "--blocksize-mode"
    )
