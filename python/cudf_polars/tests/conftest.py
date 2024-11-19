# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from dask.distributed import Client, LocalCluster


@pytest.fixture(params=[False, True], ids=["no_nulls", "nulls"], scope="session")
def with_nulls(request):
    return request.param


class DistributedClusterResource:
    """Singleton class to manage Distributed cluster and client.

    A singleton class with the purpose of managing a Distributed cluster and
    client, ensuring only one instance of each exists and lives throughout the
    entire pytest session.
    """

    _instance = None

    def __init__(self):
        _cluster = None
        _client = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self):
        if self._cluster is None:
            self._cluster = LocalCluster()
        if self._client is None:
            self._client = Client(self._cluster)

    def stop(self):
        self._client = None
        self._cluster = None


def pytest_addoption(parser):
    parser.addoption(
        "--executor",
        action="store",
        default="cudf",
        choices=("cudf", "dask-experimental"),
        help="Executor to use for GPUEngine.",
    )

    parser.addoption(
        "--dask-cluster",
        action="store_true",
        help="Executor to use for GPUEngine.",
    )


def pytest_configure(config):
    import cudf_polars.testing.asserts

    if config.getoption("--dask-cluster"):
        if config.getoption("--executor") != "dask-experimental":
            raise pytest.UsageError(
                "--dask-cluster requires --executor='dask-experimental'"
            )
        else:
            config._dask_cluster = LocalCluster()
            config._dask_client = Client()

    cudf_polars.testing.asserts.Executor = config.getoption("--executor")


def pytest_sessionstart(session):
    if (
        session.config.getoption("--dask-cluster")
        and session.config.getoption("--executor") != "dask-experimental"
    ):
        DistributedClusterResource.get_instance().start()


def pytest_sessionfinish(session):
    DistributedClusterResource.get_instance().stop()
