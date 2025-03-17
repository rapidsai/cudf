# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest

DISTRIBUTED_CLUSTER_KEY = pytest.StashKey[dict]()


@pytest.fixture(params=[False, True], ids=["no_nulls", "nulls"], scope="session")
def with_nulls(request):
    return request.param


def pytest_addoption(parser):
    parser.addoption(
        "--executor",
        action="store",
        default="pylibcudf",
        choices=("pylibcudf", "dask-experimental"),
        help="Executor to use for GPUEngine.",
    )

    parser.addoption(
        "--dask-cluster",
        action="store_true",
        help="Executor to use for GPUEngine.",
    )

    parser.addoption(
        "--rapidsmp",
        action="store_true",
        help="Use LocalRMPCluster.",
    )


def pytest_configure(config):
    import cudf_polars.testing.asserts

    if config.getoption("--rapidsmp") and not config.getoption("--dask-cluster"):
        raise pytest.UsageError("--rapidsmp requires --dask-cluster'")

    if (
        config.getoption("--dask-cluster")
        and config.getoption("--executor") != "dask-experimental"
    ):
        raise pytest.UsageError(
            "--dask-cluster requires --executor='dask-experimental'"
        )

    cudf_polars.testing.asserts.Executor = config.getoption("--executor")


def pytest_sessionstart(session):
    if (
        session.config.getoption("--dask-cluster")
        and session.config.getoption("--executor") == "dask-experimental"
    ):
        from dask import config
        from dask.distributed import Client

        # Avoid "Sending large graph of size ..." warnings
        # (We expect these for tests using literal/random arrays)
        config.set({"distributed.admin.large-graph-warning-threshold": "20MB"})

        n_workers = int(os.environ.get("CUDF_POLARS_NUM_WORKERS", "1"))
        if session.config.getoption("--rapidsmp"):
            from rapidsmp.integrations.dask import (
                LocalRMPCluster,
                bootstrap_dask_cluster,
            )

            cluster = LocalRMPCluster(n_workers=n_workers)
            client = Client(cluster)
            client.wait_for_workers(n_workers)
            bootstrap_dask_cluster(client, enable_statistics=False)
        else:
            from dask_cuda import LocalCUDACluster

            cluster = LocalCUDACluster(n_workers=n_workers)
            client = Client(cluster)
        session.stash[DISTRIBUTED_CLUSTER_KEY] = {"cluster": cluster, "client": client}


def pytest_sessionfinish(session):
    if DISTRIBUTED_CLUSTER_KEY in session.stash:
        cluster_info = session.stash[DISTRIBUTED_CLUSTER_KEY]
        client = cluster_info.get("client")
        cluster = cluster_info.get("cluster")
        if client is not None:
            client.shutdown()
        if cluster is not None:
            cluster.close()
