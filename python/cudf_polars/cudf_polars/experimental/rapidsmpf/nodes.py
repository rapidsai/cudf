# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core node definitions for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.core.spillable_messages import SpillableMessages
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Cache, Empty, Filter, Projection
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    Metadata,
    empty_table_chunk,
    make_spill_function,
    process_children,
    shutdown_on_error,
)

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


@define_py_node()
async def default_node_single(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    *,
    preserve_partitioning: bool = False,
) -> None:
    """
    Single-channel default node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    ch_in
        The input ChannelPair.
    preserve_partitioning
        Whether to preserve the partitioning metadata of the input chunks.

    Notes
    -----
    Chunks are processed in the order they are received.
    """
    async with shutdown_on_error(
        context, ch_in.metadata, ch_in.data, ch_out.metadata, ch_out.data
    ):
        # Recv/send metadata.
        metadata_in = await ch_in.recv_metadata(context)
        metadata_out = Metadata(
            metadata_in.count,
            partitioned_on=metadata_in.partitioned_on if preserve_partitioning else (),
            duplicated=metadata_in.duplicated,
        )
        await ch_out.send_metadata(context, metadata_out)

        # Recv/send data.
        while (msg := await ch_in.data.recv(context)) is not None:
            chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            seq_num = msg.sequence_number
            df = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                DataFrame.from_table(
                    chunk.table_view(),
                    list(ir.children[0].schema.keys()),
                    list(ir.children[0].schema.values()),
                    chunk.stream,
                ),
                context=ir_context,
            )
            chunk = TableChunk.from_pylibcudf_table(
                df.table, chunk.stream, exclusive_view=True
            )
            await ch_out.data.send(context, Message(seq_num, chunk))

        await ch_out.data.drain(context)


@define_py_node()
async def default_node_multi(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    chs_in: tuple[ChannelPair, ...],
    *,
    partitioning_index: int | None = None,
) -> None:
    """
    Pointwise node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    chs_in
        Tuple of input ChannelPairs.
    partitioning_index
        Index of the input channel to preserve partitioning information for.
        If None, no partitioning information is preserved.
    """
    async with shutdown_on_error(
        context,
        *[ch.metadata for ch in chs_in],
        ch_out.metadata,
        *[ch.data for ch in chs_in],
        ch_out.data,
    ):
        # Merge and forward basic metadata.
        metadata = Metadata(1)
        for idx, ch_in in enumerate(chs_in):
            md_child = await ch_in.recv_metadata(context)
            metadata.count = max(md_child.count, metadata.count)
            metadata.duplicated = metadata.duplicated and md_child.duplicated
            if idx == partitioning_index:
                metadata.partitioned_on = md_child.partitioned_on
        await ch_out.send_metadata(context, metadata)

        seq_num = 0
        n_children = len(chs_in)
        finished_channels: set[int] = set()
        # Store TableChunk objects to keep data alive and prevent use-after-free
        # with stream-ordered allocations
        ready_chunks: list[TableChunk | None] = [None] * n_children
        chunk_count: list[int] = [0] * n_children

        # Recv/send data.
        while True:
            # Receive from all non-finished channels
            for ch_idx, (ch_in, _child) in enumerate(
                zip(chs_in, ir.children, strict=True)
            ):
                if ch_idx in finished_channels:
                    continue  # This channel already finished, reuse its data

                msg = await ch_in.data.recv(context)
                if msg is None:
                    # Channel finished - keep its last chunk for reuse
                    finished_channels.add(ch_idx)
                else:
                    # Store the new chunk (replacing previous if any)
                    ready_chunks[ch_idx] = TableChunk.from_message(msg)
                    chunk_count[ch_idx] += 1
                assert ready_chunks[ch_idx] is not None, (
                    f"Channel {ch_idx} has no data after receive loop."
                )

            # If all channels finished, we're done
            if len(finished_channels) == n_children:
                break

            # Convert chunks to DataFrames right before evaluation
            # All chunks are guaranteed to be non-None by the assertion above
            assert all(chunk is not None for chunk in ready_chunks), (
                "All chunks must be non-None"
            )
            # Ensure all table chunks are unspilled and available.
            ready_chunks = [
                chunk.make_available_and_spill(context.br(), allow_overbooking=True)
                for chunk in cast(list[TableChunk], ready_chunks)
            ]
            dfs = [
                DataFrame.from_table(
                    chunk.table_view(),  # type: ignore[union-attr]
                    list(child.schema.keys()),
                    list(child.schema.values()),
                    chunk.stream,  # type: ignore[union-attr]
                )
                for chunk, child in zip(ready_chunks, ir.children, strict=True)
            ]

            # Evaluate the IR node with current chunks
            df = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                *dfs,
                context=ir_context,
            )
            await ch_out.data.send(
                context,
                Message(
                    seq_num,
                    TableChunk.from_pylibcudf_table(
                        df.table,
                        df.stream,
                        exclusive_view=True,
                    ),
                ),
            )
            seq_num += 1

        # Drain the output channel
        await ch_out.data.drain(context)


@define_py_node()
async def fanout_node_bounded(
    context: Context,
    ch_in: ChannelPair,
    *chs_out: ChannelPair,
) -> None:
    """
    Bounded fanout node for rapidsmpf.

    Each chunk is broadcasted to all output channels
    as it arrives.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ch_in
        The input ChannelPair.
    chs_out
        The output ChannelPairs.
    """
    # TODO: Use rapidsmpf fanout node once available.
    # See: https://github.com/rapidsai/rapidsmpf/issues/560
    async with shutdown_on_error(
        context,
        ch_in.metadata,
        ch_in.data,
        *[ch.metadata for ch in chs_out],
        *[ch.data for ch in chs_out],
    ):
        # Forward metadata to all outputs.
        metadata = await ch_in.recv_metadata(context)
        await asyncio.gather(*(ch.send_metadata(context, metadata) for ch in chs_out))

        while (msg := await ch_in.data.recv(context)) is not None:
            table_chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            seq_num = msg.sequence_number
            for ch_out in chs_out:
                await ch_out.data.send(
                    context,
                    Message(
                        seq_num,
                        TableChunk.from_pylibcudf_table(
                            table_chunk.table_view(),
                            table_chunk.stream,
                            exclusive_view=False,
                        ),
                    ),
                )

        await asyncio.gather(*(ch.data.drain(context) for ch in chs_out))


@define_py_node()
async def fanout_node_unbounded(
    context: Context,
    ch_in: ChannelPair,
    *chs_out: ChannelPair,
) -> None:
    """
    Unbounded fanout node for rapidsmpf with spilling support.

    Broadcasts chunks from input to all output channels. This is called
    "unbounded" because it handles the case where one channel may consume
    all data before another channel consumes any data.

    The implementation uses adaptive sending with spillable buffers:
    - Maintains a spillable FIFO buffer for each output channel
    - Messages are buffered in host memory (spillable to disk)
    - Sends to all channels concurrently
    - Receives next chunk as soon as any channel makes progress
    - Efficient for both balanced and imbalanced consumption patterns

    Parameters
    ----------
    context
        The rapidsmpf context.
    ch_in
        The input ChannelPair.
    chs_out
        The output ChannelPairs.
    """
    # TODO: Use rapidsmpf fanout node once available.
    # See: https://github.com/rapidsai/rapidsmpf/issues/560
    async with shutdown_on_error(
        context,
        ch_in.metadata,
        ch_in.data,
        *[ch.metadata for ch in chs_out],
        *[ch.data for ch in chs_out],
    ):
        # Forward metadata to all outputs.
        metadata = await ch_in.recv_metadata(context)
        await asyncio.gather(*(ch.send_metadata(context, metadata) for ch in chs_out))

        # Spillable FIFO buffer for each output channel
        output_buffers: list[SpillableMessages] = [SpillableMessages() for _ in chs_out]
        num_outputs = len(chs_out)

        # Track message IDs in FIFO order for each output buffer
        buffer_ids: list[list[int]] = [[] for _ in chs_out]

        # Register a single spill function for all buffers
        # This ensures global FIFO ordering when spilling across all outputs
        spill_func_id = context.br().spill_manager.add_spill_function(
            make_spill_function(output_buffers, context), priority=0
        )

        try:
            # Track active send/drain tasks for each output
            active_tasks: dict[int, asyncio.Task] = {}

            # Track which outputs need to be drained (set when no more input)
            needs_drain: set[int] = set()

            # Receive task
            recv_task: asyncio.Task | None = asyncio.create_task(
                ch_in.data.recv(context)
            )

            # Flag to indicate we should start a new receive (for backpressure)
            can_receive: bool = True

            async def send_one_from_buffer(idx: int) -> None:
                """
                Send one buffered message for output idx.

                The message remains in host memory (spillable) through the channel.
                The downstream consumer will call make_available() when needed.
                """
                if buffer_ids[idx]:
                    mid = buffer_ids[idx].pop(0)
                    msg = output_buffers[idx].extract(mid=mid)
                    await chs_out[idx].data.send(context, msg)

            async def drain_output(idx: int) -> None:
                """Drain output channel idx."""
                await chs_out[idx].data.drain(context)

            # Main loop: coordinate receiving, sending, and draining
            while (
                recv_task is not None or active_tasks or any(buffer_ids) or needs_drain
            ):
                # Collect all currently active tasks
                tasks_to_wait = list(active_tasks.values())
                # Only include recv_task if we're allowed to receive
                if recv_task is not None and can_receive:
                    tasks_to_wait.append(recv_task)

                # Start new tasks for outputs with work to do
                for idx in range(len(chs_out)):
                    if idx not in active_tasks:
                        if buffer_ids[idx]:
                            # Send next buffered message
                            task = asyncio.create_task(send_one_from_buffer(idx))
                            active_tasks[idx] = task
                            tasks_to_wait.append(task)
                        elif idx in needs_drain:
                            # Buffer empty and no more input - drain this output
                            task = asyncio.create_task(drain_output(idx))
                            active_tasks[idx] = task
                            tasks_to_wait.append(task)
                            needs_drain.discard(idx)

                # If nothing to wait for, we're done
                if not tasks_to_wait:
                    break

                # Wait for ANY task to complete
                done, _ = await asyncio.wait(
                    tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
                )

                # Process completed tasks
                for task in done:
                    if task is recv_task:
                        # Receive completed
                        msg = task.result()
                        if msg is None:
                            # End of input - mark all outputs as needing drain
                            recv_task = None
                            needs_drain.update(range(len(chs_out)))
                        else:
                            # Determine where to copy based on:
                            # 1. Current message location (avoid unnecessary transfers)
                            # 2. Available memory (avoid OOM)
                            content_desc = msg.get_content_description()
                            device_size = content_desc.content_sizes.get(
                                MemoryType.DEVICE, 0
                            )
                            copy_cost = msg.copy_cost()

                            # Check if we have enough device memory for all copies
                            # We need (num_outputs - 1) copies since last one reuses original
                            num_copies = num_outputs - 1
                            total_copy_cost = copy_cost * num_copies
                            available_device_mem = context.br().memory_available(
                                MemoryType.DEVICE
                            )

                            # Decide target memory:
                            # Use device ONLY if message is in device AND we have sufficient headroom.
                            # TODO: Use further information about the downstream operations to make
                            # a more informed decision.
                            required_headroom = total_copy_cost * 2
                            if (
                                device_size > 0
                                and available_device_mem >= required_headroom
                            ):
                                # Use reserve_device_memory_and_spill to automatically trigger spilling
                                # if needed to make room for the copy
                                memory_reservation = (
                                    context.br().reserve_device_memory_and_spill(
                                        total_copy_cost,
                                        allow_overbooking=True,
                                    )
                                )
                            else:
                                # Use host memory for buffering - much safer
                                # Downstream consumers will make_available() when they need device memory
                                memory_reservation, _ = context.br().reserve(
                                    MemoryType.HOST,
                                    total_copy_cost,
                                    allow_overbooking=True,
                                )

                            # Copy message for each output buffer
                            # Copies are spillable and allow downstream consumers
                            # to control device memory allocation
                            for idx, sm in enumerate(output_buffers):
                                if idx < num_outputs - 1:
                                    # Copy to target memory and insert into spillable buffer
                                    mid = sm.insert(msg.copy(memory_reservation))
                                else:
                                    # Optimization: reuse the original message for last output
                                    # (no copy needed)
                                    mid = sm.insert(msg)
                                buffer_ids[idx].append(mid)

                            # Don't receive next chunk until at least one send completes
                            can_receive = False
                            recv_task = asyncio.create_task(ch_in.data.recv(context))
                    else:
                        # Must be a send or drain task - find which output and remove it
                        for idx, at in list(active_tasks.items()):
                            if at is task:
                                del active_tasks[idx]
                                # A send completed - allow receiving again
                                can_receive = True
                                break

        finally:
            # Clean up spill function registration
            context.br().spill_manager.remove_spill_function(spill_func_id)


@generate_ir_sub_network.register(IR)
def _(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Default generate_ir_sub_network logic.
    # Use simple pointwise node.

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    if len(ir.children) == 1:
        # Single-channel default node
        preserve_partitioning = isinstance(
            # TODO: We don't need to worry about
            # non-pointwise Filter operations here,
            # because the lowering stage would have
            # collapsed to one partition anyway.
            ir,
            (Cache, Projection, Filter),
        )
        nodes[ir] = [
            default_node_single(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[ir.children[0]].reserve_output_slot(),
                preserve_partitioning=preserve_partitioning,
            )
        ]
    else:
        # Multi-channel default node
        nodes[ir] = [
            default_node_multi(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                tuple(channels[c].reserve_output_slot() for c in ir.children),
            )
        ]

    return nodes, channels


@define_py_node()
async def empty_node(
    context: Context,
    ir: Empty,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
) -> None:
    """
    Empty node for rapidsmpf - produces a single empty chunk.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Empty node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    """
    async with shutdown_on_error(context, ch_out.metadata, ch_out.data):
        # Send metadata indicating a single empty chunk
        await ch_out.send_metadata(context, Metadata(1, duplicated=True))

        # Evaluate the IR node to create an empty DataFrame
        df: DataFrame = ir.do_evaluate(*ir._non_child_args, context=ir_context)

        # Return the output chunk (empty but with correct schema)
        chunk = TableChunk.from_pylibcudf_table(
            df.table, df.stream, exclusive_view=True
        )
        await ch_out.data.send(context, Message(0, chunk))

        await ch_out.data.drain(context)


@generate_ir_sub_network.register(Empty)
def _(
    ir: Empty, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate network for Empty node - produces one empty chunk."""
    context = rec.state["context"]
    ir_context = rec.state["ir_context"]
    channels: dict[IR, ChannelManager] = {ir: ChannelManager(rec.state["context"])}
    nodes: dict[IR, list[Any]] = {
        ir: [empty_node(context, ir, ir_context, channels[ir].reserve_input_slot())]
    }
    return nodes, channels


def generate_ir_sub_network_wrapper(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """
    Generate a sub-network for the RapidsMPF streaming runtime.

    Parameters
    ----------
    ir
        The IR node.
    rec
        Recursive SubNetGenerator callable.

    Returns
    -------
    nodes
        Dictionary mapping each IR node to its list of streaming-network node(s).
    channels
        Dictionary mapping between each IR node and its
        corresponding streaming-network output ChannelManager.
    """
    nodes, channels = generate_ir_sub_network(ir, rec)

    # Check if this node needs fanout
    if (fanout_info := rec.state["fanout_nodes"].get(ir)) is not None:
        count = fanout_info.num_consumers
        manager = ChannelManager(rec.state["context"], count=count)
        fanout_node: Any
        if fanout_info.unbounded:
            fanout_node = fanout_node_unbounded(
                rec.state["context"],
                channels[ir].reserve_output_slot(),
                *[manager.reserve_input_slot() for _ in range(count)],
            )
        else:  # "bounded"
            fanout_node = fanout_node_bounded(
                rec.state["context"],
                channels[ir].reserve_output_slot(),
                *[manager.reserve_input_slot() for _ in range(count)],
            )
        nodes[ir].append(fanout_node)
        channels[ir] = manager
    return nodes, channels


@define_py_node()
async def metadata_feeder_node(
    context: Context,
    channel: ChannelPair,
    metadata: Metadata,
) -> None:
    """
    Feed metadata to a channel pair.

    Parameters
    ----------
    context
        The rapidsmpf context.
    channel
        The channel pair.
    metadata
        The metadata to feed.
    """
    async with shutdown_on_error(context, channel.metadata, channel.data):
        await channel.send_metadata(context, metadata)


@define_py_node()
async def metadata_drain_node(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_in: ChannelPair,
    ch_out: Any,
    metadata_collector: list[Metadata] | None,
) -> None:
    """
    Drain metadata and forward data to a single channel.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node.
    ir_context
        The execution context for the IR node.
    ch_in
        The input ChannelPair (with metadata and data channels).
    ch_out
        The output data channel.
    metadata_collector
        The list to collect the final metadata.
        This list will be mutated when the network is executed.
        If None, metadata will not be collected.
    """
    async with shutdown_on_error(context, ch_in.metadata, ch_in.data, ch_out):
        # Drain metadata channel (we don't need it after this point)
        metadata = await ch_in.recv_metadata(context)
        send_empty = metadata.duplicated and context.comm().rank != 0
        if metadata_collector is not None:
            metadata_collector.append(metadata)

        # Forward non-duplicated data messages
        while (msg := await ch_in.data.recv(context)) is not None:
            if not send_empty:
                await ch_out.send(context, msg)

        # Send empty data if needed
        if send_empty:
            stream = ir_context.get_cuda_stream()
            await ch_out.send(
                context, Message(0, empty_table_chunk(ir, context, stream))
            )

        await ch_out.drain(context)
