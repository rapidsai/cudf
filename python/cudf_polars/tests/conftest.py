# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
        default="pylibcudf",
        choices=("pylibcudf", "dask-experimental"),
        help="Executor to use for GPUEngine.",
    )


def pytest_configure(config):
    import cudf_polars.testing.asserts

    cudf_polars.testing.asserts.Executor = config.getoption("--executor")
