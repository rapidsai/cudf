# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF spilling functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rapidsmpf.communicator.single import new_communicator as single_process_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource, LimitAvailableMemory
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.spillable_messages import SpillableMessages
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc
import rmm.mr

from cudf_polars.experimental.rapidsmpf.utils import (
    make_spill_function,
    opaque_reservation,
)

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream


def create_test_table(nbytes: int, stream: Stream) -> plc.Table:
    """Create a test table with specified size in bytes."""
    assert nbytes % 4 == 0, "nbytes must be divisible by 4 for float32"
    # Create a simple table with one column of random float32 data
    num_elements = nbytes // 4
    data = np.random.random(num_elements).astype(np.float32)
    # mypy doesn't recognize pylibcudf's from_array signature correctly
    return plc.Table([plc.Column.from_array(data, stream=stream)])  # type: ignore[call-arg]


def test_make_spill_function(local_context: Context) -> None:
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
    stream = local_context.get_stream_from_pool()
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
    spill_func = make_spill_function(buffers, local_context)
    func_id = local_context.br().spill_manager.add_spill_function(
        spill_func, priority=0
    )

    try:
        # Manually trigger spilling of 3MB
        # Expected: Buffer 1 (longest) should spill newest messages first
        amount_to_spill = 3 * 1024 * 1024
        actual_spilled = local_context.br().spill_manager.spill(amount_to_spill)

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
        res, _ = local_context.br().reserve(
            MemoryType.DEVICE, cost, allow_overbooking=True
        )
        chunk_available = chunk.make_available(res)

        assert chunk_available.is_available()
        # Verify we got a valid table back
        assert chunk_available.table_view().num_rows() > 0

    finally:
        local_context.br().spill_manager.remove_spill_function(func_id)


def test_opaque_reservation() -> None:
    options = Options(get_environment_variables())
    comm = single_process_comm(options)
    _original_mr = rmm.mr.get_current_device_resource()
    mr = RmmResourceAdaptor(_original_mr)
    rmm.mr.set_current_device_resource(mr)

    # Set a 100MB limit
    limit = 100 * 1024 * 1024
    memory_available = {MemoryType.DEVICE: LimitAvailableMemory(mr, limit=limit)}
    br = BufferResource(mr, memory_available=memory_available)
    context = Context(comm, br, options)
    stream = context.get_stream_from_pool()

    # Create spillable data container
    spillable = SpillableMessages()

    # Allocate 80MB of spillable data (8 x 10MB chunks)
    chunk_size = 10 * 1024 * 1024
    message_ids = []
    for i in range(8):
        table = create_test_table(chunk_size, stream)
        chunk = TableChunk.from_pylibcudf_table(table, stream, exclusive_view=True)
        msg = Message(i, chunk)
        mid = spillable.insert(msg)
        message_ids.append(mid)

    # Register spill function so rapidsmpf can spill our data
    spill_func = make_spill_function([spillable], context)
    func_id = br.spill_manager.add_spill_function(spill_func, priority=0)

    try:
        # Verify all data is on device initially
        descs_before = spillable.get_content_descriptions()
        for mid in message_ids:
            assert descs_before[mid].content_sizes[MemoryType.DEVICE] > 0
            assert descs_before[mid].content_sizes[MemoryType.HOST] == 0

        # Now request a 50MB reservation - this should trigger spilling
        # since we only have ~20MB available (100MB limit - 80MB used)
        reserve_size = 50 * 1024 * 1024

        available_before = br.memory_available(MemoryType.DEVICE)
        reserved_before = br.memory_reserved(MemoryType.DEVICE)
        print("\nBefore reservation:")
        print(f"  memory_available: {available_before / (1024 * 1024):.2f}MB")
        print(f"  memory_reserved: {reserved_before / (1024 * 1024):.2f}MB")
        print(
            f"  current_allocated (computed): {(limit - available_before) / (1024 * 1024):.2f}MB"
        )

        with opaque_reservation(context, reserve_size) as reservation:
            assert reservation.size == reserve_size

            available_after = br.memory_available(MemoryType.DEVICE)
            reserved_after = br.memory_reserved(MemoryType.DEVICE)
            print("\nAfter reservation (inside context):")
            print(f"  reservation.size: {reservation.size / (1024 * 1024):.2f}MB")
            print(f"  memory_available: {available_after / (1024 * 1024):.2f}MB")
            print(f"  memory_reserved: {reserved_after / (1024 * 1024):.2f}MB")
            print(
                f"  current_allocated (computed): {(limit - available_after) / (1024 * 1024):.2f}MB"
            )

            # Check that some data was spilled to host
            descs_after = spillable.get_content_descriptions()
            spilled_count = sum(
                1
                for mid in message_ids
                if descs_after[mid].content_sizes[MemoryType.HOST] > 0
            )
            spilled_bytes = sum(
                descs_after[mid].content_sizes[MemoryType.HOST] for mid in message_ids
            )
            device_bytes = sum(
                descs_after[mid].content_sizes[MemoryType.DEVICE] for mid in message_ids
            )
            print("\nSpilling results:")
            print(f"  Chunks spilled: {spilled_count} of {len(message_ids)}")
            print(f"  Bytes on HOST: {spilled_bytes / (1024 * 1024):.2f}MB")
            print(f"  Bytes on DEVICE: {device_bytes / (1024 * 1024):.2f}MB")

            # We need to spill at least 30MB (50MB - 20MB available)
            # That's at least 3 chunks of 10MB each
            assert spilled_count >= 3
            assert available_after >= reserve_size

    finally:
        br.spill_manager.remove_spill_function(func_id)
        rmm.mr.set_current_device_resource(_original_mr)
