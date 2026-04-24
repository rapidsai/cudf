# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util

import pytest

import cudf_polars.callback


@pytest.fixture(params=[False, True], ids=["no_nulls", "nulls"], scope="session")
def with_nulls(request):
    return request.param


@pytest.fixture
def clear_memory_resource_cache():
    """
    Clear the cudf_polars.callback.default_memory_resource cache before and after a test.

    This function caches memory resources for the duration of the process. Any test that
    creates a pool (e.g. ``CudaAsyncMemoryResource``) should use this fixture to ensure that
    the pool is freed after the test.
    """
    cudf_polars.callback.default_memory_resource.cache_clear()
    yield
    cudf_polars.callback.default_memory_resource.cache_clear()


@pytest.fixture
def using_rapidsmpf():
    return cudf_polars.testing.asserts.DEFAULT_RUNTIME == "rapidsmpf"


def pytest_addoption(parser):
    parser.addoption(
        "--executor",
        action="store",
        default="streaming",
        choices=("in-memory", "streaming"),
        help="Executor to use for GPUEngine.",
    )

    parser.addoption(
        "--runtime",
        action="store",
        default="tasks",
        choices=("tasks", "rapidsmpf"),
        help="Runtime to use for the 'streaming' executor.",
    )

    parser.addoption(
        "--cluster",
        action="store",
        default="single",
        choices=("single", "distributed"),
        help="Cluster to use for 'streaming' executor.",
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


def pytest_configure(config):
    import cudf_polars.testing.asserts

    # Ray's internal subprocess management leaks `/dev/null` file handles, and
    # distributed's shutdown leaves unclosed sockets. Under Python 3.14 +
    # pytest 9, these surface as unraisable `ResourceWarning`s and — combined
    # with `filterwarnings = ["error", ...]` in pyproject.toml — fail
    # otherwise-unrelated tests when the GC finalizer happens to fire during
    # them. With `pytest-xdist --dist=worksteal`, the leak can land in any
    # test that shares a worker with a ray/dask test, so the suppression must
    # apply globally rather than per-module.
    config.addinivalue_line("filterwarnings", "ignore::ResourceWarning")

    if (
        config.getoption("--cluster") == "distributed"
        and config.getoption("--executor") != "streaming"
    ):
        raise pytest.UsageError("Distributed cluster requires --executor='streaming'")

    if config.getoption("--runtime") == "rapidsmpf":
        if config.getoption("--executor") == "in-memory":
            raise pytest.UsageError("Rapidsmpf runtime requires --executor='streaming'")

        if importlib.util.find_spec("rapidsmpf") is None:
            raise pytest.UsageError(
                "Rapidsmpf runtime requires the 'rapidsmpf' package"
            )

    cudf_polars.testing.asserts.DEFAULT_EXECUTOR = config.getoption("--executor")
    cudf_polars.testing.asserts.DEFAULT_RUNTIME = config.getoption("--runtime")
    cudf_polars.testing.asserts.DEFAULT_CLUSTER = config.getoption("--cluster")
    cudf_polars.testing.asserts.DEFAULT_BLOCKSIZE_MODE = config.getoption(
        "--blocksize-mode"
    )
