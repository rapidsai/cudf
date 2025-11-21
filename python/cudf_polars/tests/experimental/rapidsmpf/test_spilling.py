# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF spilling functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from rapidsmpf.communicator.single import new_communicator as single_process_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.spillable_messages import SpillableMessages
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc
import rmm.mr

from cudf_polars.experimental.rapidsmpf.utils import make_spill_function

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream


@pytest.fixture
def context() -> Context:
    """Fixture to create a single-GPU streaming context for testing."""
    options = Options(get_environment_variables())
    comm = single_process_comm(options)
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    br = BufferResource(mr)
    return Context(comm, br, options)


def create_test_table(nbytes: int, stream: Stream) -> plc.Table:
    """Create a test table with specified size in bytes."""
    assert nbytes % 4 == 0, "nbytes must be divisible by 4 for float32"
    # Create a simple table with one column of random float32 data
    num_elements = nbytes // 4
    data = np.random.random(num_elements).astype(np.float32)
    # mypy doesn't recognize pylibcudf's from_array signature correctly
    return plc.Table([plc.Column.from_array(data, stream=stream)])  # type: ignore[call-arg]


def test_make_spill_function(context: Context) -> None:
    """Test that spilling prioritizes longest queues and newest messages."""

    # Create 3 spillable message containers simulating fanout buffers
    # Buffer 0: Fast consumer (2 messages)
    # Buffer 1: Slow consumer (5 messages) <- should spill from here first
    # Buffer 2: Medium consumer (3 messages)
    buffers = [SpillableMessages() for _ in range(3)]
    messages_per_buffer = [2, 5, 3]

    # Track message IDs for each buffer
    message_ids: dict[int, list[int]] = {}

    # Populate buffers with messages
    stream = context.get_stream_from_pool()
    for buffer_idx, (sm, count) in enumerate(
        zip(buffers, messages_per_buffer, strict=False)
    ):
        message_ids[buffer_idx] = []
        for msg_idx in range(count):
            # Create 1MB messages
            table = create_test_table(1024 * 1024, stream)
            chunk = TableChunk.from_pylibcudf_table(table, stream, exclusive_view=True)
            msg = Message(msg_idx, chunk)
            mid = sm.insert(msg)
            message_ids[buffer_idx].append(mid)

    # Register spill function
    spill_func = make_spill_function(buffers, context)
    func_id = context.br().spill_manager.add_spill_function(spill_func, priority=0)

    try:
        # Manually trigger spilling of 3MB
        # Expected: Buffer 1 (longest) should spill newest messages first
        amount_to_spill = 3 * 1024 * 1024
        actual_spilled = context.br().spill_manager.spill(amount_to_spill)

        # Allow some tolerance
        assert actual_spilled >= amount_to_spill * 0.95

        # Verify Buffer 1 (longest queue): newest 3 messages should be spilled
        buffer_1_descs = buffers[1].get_content_descriptions()
        for i in range(3, 5):  # Messages 3, 4 (newest)
            mid = message_ids[1][i]
            desc = buffer_1_descs[mid]
            # Should be in HOST memory (spilled)
            assert desc.content_sizes[MemoryType.HOST] > 0
            assert desc.content_sizes[MemoryType.DEVICE] == 0

        # Buffer 1: oldest messages should still be in device
        for i in range(2):  # Messages 0, 1 (oldest)
            mid = message_ids[1][i]
            desc = buffer_1_descs[mid]
            # Should still be in DEVICE memory
            assert desc.content_sizes[MemoryType.DEVICE] > 0
            assert desc.content_sizes[MemoryType.HOST] == 0

        # Buffer 0 (shortest queue): all messages should still be on device
        buffer_0_descs = buffers[0].get_content_descriptions()
        for mid in message_ids[0]:
            desc = buffer_0_descs[mid]
            assert desc.content_sizes[MemoryType.DEVICE] > 0
            assert desc.content_sizes[MemoryType.HOST] == 0

        # Verify we can extract and make available a spilled message
        spilled_mid = message_ids[1][4]  # Newest message from longest queue
        spilled_msg = buffers[1].extract(mid=spilled_mid)

        chunk = TableChunk.from_message(spilled_msg)
        assert not chunk.is_available()  # Should be on host

        # Make it available should bring it back to device
        cost = chunk.make_available_cost()
        assert cost > 0
        res, _ = context.br().reserve(MemoryType.DEVICE, cost, allow_overbooking=True)
        chunk_available = chunk.make_available(res)

        assert chunk_available.is_available()
        # Verify we got a valid table back
        assert chunk_available.table_view().num_rows() > 0

    finally:
        context.br().spill_manager.remove_spill_function(func_id)
