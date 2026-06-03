# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import rmm.mr
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.context import Context

if TYPE_CHECKING:
    from collections.abc import Generator

    from rapidsmpf.communicator.communicator import Communicator

# Import fixtures (comm, device_mr, etc.) from rapidsmpf's test conftest
# if available (conda installs include tests; wheel installs may not).
try:
    import rapidsmpf.tests.conftest  # noqa: F401

    _HAS_RAPIDSMPF_TEST_FIXTURES = True
    pytest_plugins = ["rapidsmpf.tests.conftest"]
except (ImportError, ModuleNotFoundError):
    _HAS_RAPIDSMPF_TEST_FIXTURES = False


@pytest.fixture
def context(comm: Communicator) -> Generator[Context, None, None]:
    """
    Fixture to get a streaming context.
    """
    options = Options(get_environment_variables())
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    br = BufferResource(mr)

    with Context(comm.logger, br, options) as ctx:
        yield ctx


if not _HAS_RAPIDSMPF_TEST_FIXTURES:

    @pytest.fixture(params=["mpi", "ucxx"])
    def comm(request):
        """Fallback comm fixture that skips when rapidsmpf test infra is unavailable."""
        pytest.skip("rapidsmpf test fixtures not installed")

    @pytest.fixture
    def stream():
        """Fallback stream fixture that skips when rapidsmpf test infra is unavailable."""
        pytest.skip("rapidsmpf test fixtures not installed")
