# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for rapidsmpf tests."""

from __future__ import annotations

import pytest

# Skip all tests in this directory if rapidsmpf is not available
pytest.importorskip("rapidsmpf")

from rapidsmpf.communicator.single import new_communicator as single_process_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.context import Context

import rmm.mr


@pytest.fixture
def local_context() -> Context:
    """Fixture to create a single-GPU streaming context for testing."""
    options = Options(get_environment_variables())
    comm = single_process_comm(options)
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    br = BufferResource(mr)
    return Context(comm, br, options)
