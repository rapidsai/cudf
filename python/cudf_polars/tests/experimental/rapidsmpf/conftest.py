# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for rapidsmpf tests."""

from __future__ import annotations

import pytest

# Skip all tests in this directory if rapidsmpf is not available
pytest.importorskip("rapidsmpf")

from typing import TYPE_CHECKING

from rapidsmpf.communicator.single import new_communicator as single_process_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.context import Context

import rmm.mr

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator


@pytest.fixture
def local_comm() -> Communicator:
    """Fixture to create a single-GPU communicator for testing."""
    options = Options(get_environment_variables())
    return single_process_comm(options, ProgressThread())


@pytest.fixture
def local_context(local_comm: Communicator) -> Context:
    """Fixture to create a single-GPU streaming context for testing."""
    options = Options(get_environment_variables())
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    br = BufferResource(mr)
    return Context(local_comm.logger, br, options)
