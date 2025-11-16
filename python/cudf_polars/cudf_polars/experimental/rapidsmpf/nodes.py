# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core node definitions for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Empty
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
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

    Notes
    -----
    Chunks are processed in the order they are received.
    """
    async with shutdown_on_error(context, ch_in.data, ch_out.data):
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
        The output ChannelPair (metadata already sent).
    chs_in
        Tuple of input ChannelPairs (metadata already received).

    Notes
    -----
    Input chunks must be aligned for evaluation. Messages from each input
    channel are assumed to arrive in sequence number order, so we only need
    to hold one chunk per channel at a time.
    """
    async with shutdown_on_error(context, *[ch.data for ch in chs_in], ch_out.data):
        seq_num = 0
        n_children = len(chs_in)
        finished_channels: set[int] = set()
        # Store TableChunk objects to keep data alive and prevent use-after-free
        # with stream-ordered allocations
        ready_chunks: list[TableChunk | None] = [None] * n_children
        chunk_count: list[int] = [0] * n_children

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
    async with shutdown_on_error(context, ch_in.data, *[ch.data for ch in chs_out]):
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
    Unbounded fanout node for rapidsmpf.

    Broadcasts chunks from input to all output channels. This is called
    "unbounded" because it handles the case where one channel may consume
    all data before another channel consumes any data.

    The implementation uses adaptive sending:
    - Maintains a FIFO buffer for each output channel
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
    async with shutdown_on_error(context, ch_in.data, *[ch.data for ch in chs_out]):
        # FIFO buffer for each output channel
        output_buffers: list[list[Message]] = [[] for _ in chs_out]

        # Track active send/drain tasks for each output
        active_tasks: dict[int, asyncio.Task] = {}

        # Track which outputs need to be drained (set when no more input)
        needs_drain: set[int] = set()

        # Receive task
        recv_task: asyncio.Task | None = asyncio.create_task(ch_in.data.recv(context))

        # Flag to indicate we should start a new receive (for backpressure)
        can_receive: bool = True

        async def send_one_from_buffer(idx: int) -> None:
            """Send one buffered message for output idx."""
            if output_buffers[idx]:
                msg = output_buffers[idx].pop(0)
                await chs_out[idx].data.send(context, msg)

        async def drain_output(idx: int) -> None:
            """Drain output channel idx."""
            await chs_out[idx].data.drain(context)

        # Main loop: coordinate receiving, sending, and draining
        while (
            recv_task is not None or active_tasks or any(output_buffers) or needs_drain
        ):
            # Collect all currently active tasks
            tasks_to_wait = list(active_tasks.values())
            # Only include recv_task if we're allowed to receive
            if recv_task is not None and can_receive:
                tasks_to_wait.append(recv_task)

            # Start new tasks for outputs with work to do
            for idx in range(len(chs_out)):
                if idx not in active_tasks:
                    if output_buffers[idx]:
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
                        # Add message to all output buffers
                        chunk = TableChunk.from_message(msg).make_available_and_spill(
                            context.br(), allow_overbooking=True
                        )
                        seq_num = msg.sequence_number
                        for buffer in output_buffers:
                            message = Message(
                                seq_num,
                                TableChunk.from_pylibcudf_table(
                                    chunk.table_view(),
                                    chunk.stream,
                                    exclusive_view=False,
                                ),
                            )
                            buffer.append(message)

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


@generate_ir_sub_network.register(IR)
def _(ir: IR, rec: SubNetGenerator) -> tuple[list[Any], dict[IR, ChannelManager]]:
    # Default generate_ir_sub_network logic.
    # Use simple pointwise node.

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    if len(ir.children) == 1:
        # Single-channel default node
        nodes.append(
            default_node_single(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[ir.children[0]].reserve_output_slot(),
            )
        )
    else:
        # Multi-channel default node
        nodes.append(
            default_node_multi(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                tuple(channels[c].reserve_output_slot() for c in ir.children),
            )
        )

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
    async with shutdown_on_error(context, ch_out.data):
        # Evaluate the IR node to create an empty DataFrame
        df: DataFrame = ir.do_evaluate(*ir._non_child_args, context=ir_context)

        # Return the output chunk (empty but with correct schema)
        chunk = TableChunk.from_pylibcudf_table(
            df.table, df.stream, exclusive_view=True
        )
        await ch_out.data.send(context, Message(0, chunk))

        await ch_out.data.drain(context)


@generate_ir_sub_network.register(Empty)
def _(ir: Empty, rec: SubNetGenerator) -> tuple[list[Any], dict[IR, ChannelManager]]:
    """Generate network for Empty node - produces one empty chunk."""
    context = rec.state["context"]
    ir_context = rec.state["ir_context"]
    channels: dict[IR, ChannelManager] = {ir: ChannelManager(rec.state["context"])}
    nodes: list[Any] = [
        empty_node(context, ir, ir_context, channels[ir].reserve_input_slot())
    ]
    return nodes, channels


def generate_ir_sub_network_wrapper(
    ir: IR, rec: SubNetGenerator
) -> tuple[list[Any], dict[IR, ChannelManager]]:
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
        List of streaming-network node(s) for the subgraph.
    channels
        Dictionary mapping between each IR node and its
        corresponding streaming-network output ChannelManager.
    """
    nodes, channels = generate_ir_sub_network(ir, rec)

    # Check if this node needs fanout
    if (fanout_info := rec.state["fanout_nodes"].get(ir)) is not None:
        count = fanout_info.num_consumers
        manager = ChannelManager(rec.state["context"], count=count)
        if fanout_info.unbounded:
            nodes.append(
                fanout_node_unbounded(
                    rec.state["context"],
                    channels[ir].reserve_output_slot(),
                    *[manager.reserve_input_slot() for _ in range(count)],
                )
            )
        else:  # "bounded"
            nodes.append(
                fanout_node_bounded(
                    rec.state["context"],
                    channels[ir].reserve_output_slot(),
                    *[manager.reserve_input_slot() for _ in range(count)],
                )
            )
        channels[ir] = manager
    return nodes, channels
