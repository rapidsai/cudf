# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
