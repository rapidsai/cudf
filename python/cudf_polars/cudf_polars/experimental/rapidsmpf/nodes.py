# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core node definitions for the RapidsMPF streaming engine."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.channel import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Empty
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    process_children,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


@asynccontextmanager
async def shutdown_on_error(
    ctx: Context, *channels: Channel[Any]
) -> AsyncIterator[None]:
    """
    Shutdown on error for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    channels
        The channels to shutdown.
    """
    # TODO: This probably belongs in rapidsmpf.
    try:
        yield
    except Exception:
        await asyncio.gather(*(ch.shutdown(ctx) for ch in channels))
        raise


@define_py_node()
async def default_node_single(
    ctx: Context,
    ir: IR,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
) -> None:
    """
    Single-channel default node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The IR node.
    ch_out
        The output ChannelPair.
    ch_in
        The input ChannelPair.

    Notes
    -----
    Chunks are processed in the order they are received.
    """
    async with shutdown_on_error(ctx, ch_in.data, ch_out.data):
        while (msg := await ch_in.data.recv(ctx)) is not None:
            chunk = TableChunk.from_message(msg)
            seq_num = chunk.sequence_number
            df = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                DataFrame.from_table(
                    chunk.table_view(),
                    list(ir.children[0].schema.keys()),
                    list(ir.children[0].schema.values()),
                    chunk.stream,
                ),
            )
            chunk = TableChunk.from_pylibcudf_table(
                seq_num, df.table, chunk.stream, exclusive_view=True
            )
            await ch_out.data.send(ctx, Message(chunk))

        await ch_out.data.drain(ctx)


@define_py_node()
async def default_node_multi(
    ctx: Context,
    ir: IR,
    ch_out: ChannelPair,
    chs_in: tuple[ChannelPair, ...],
    bcast_indices: list[int],
) -> None:
    """
    Pointwise node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The IR node.
    ch_out
        The output ChannelPair (metadata already sent).
    chs_in
        Tuple of input ChannelPairs (metadata already received).
    bcast_indices
        The indices of the broadcasted children.

    Notes
    -----
    Input chunks are aligned for evaluation.
    """
    # TODO: Use multiple streams
    async with shutdown_on_error(ctx, *[ch.data for ch in chs_in], ch_out.data):
        seq_num = 0
        n_children = len(chs_in)
        accepting_data = True
        finished_channels: set[int] = set()
        staged_chunks: dict[int, dict[int, DataFrame]] = {
            c: {} for c in range(n_children)
        }

        while True:
            if accepting_data:
                for ch_idx, (ch_in, child) in enumerate(
                    zip(chs_in, ir.children, strict=True)
                ):
                    if (ch_not_finished := ch_idx not in finished_channels) and (
                        msg := await ch_in.data.recv(ctx)
                    ) is not None:
                        table_chunk = TableChunk.from_message(msg)
                        if ch_idx in bcast_indices and staged_chunks[ch_idx]:
                            raise RuntimeError(
                                f"Broadcasted chunk already staged for channel {ch_idx}."
                            )
                        staged_chunks[ch_idx][table_chunk.sequence_number] = (
                            DataFrame.from_table(
                                table_chunk.table_view(),
                                list(child.schema.keys()),
                                list(child.schema.values()),
                                table_chunk.stream,
                            )
                        )
                    elif ch_not_finished:
                        finished_channels.add(ch_idx)
                        if all(
                            ch_idx in finished_channels for ch_idx in range(n_children)
                        ):
                            accepting_data = False

            if all(
                (
                    (seq_num in staged_chunks[ch_idx])
                    or (ch_idx in bcast_indices and 0 in staged_chunks[ch_idx])
                )
                for ch_idx in range(n_children)
            ):
                # Ready to produce the output chunk for seq_num.
                # Evaluate and send.
                df = await asyncio.to_thread(
                    ir.do_evaluate,
                    *ir._non_child_args,
                    *[
                        (
                            staged_chunks[ch_idx][0]
                            if ch_idx in bcast_indices
                            else staged_chunks[ch_idx].pop(seq_num)
                        )
                        for ch_idx in range(n_children)
                    ],
                )
                await ch_out.data.send(
                    ctx,
                    Message(
                        TableChunk.from_pylibcudf_table(
                            seq_num,
                            df.table,
                            DEFAULT_STREAM,
                            exclusive_view=True,
                        )
                    ),
                )
                seq_num += 1
            elif not accepting_data:
                if any(
                    staged_chunks[ch_idx]
                    for ch_idx in range(n_children)
                    if ch_idx not in bcast_indices
                ):
                    raise RuntimeError(
                        f"Leftover data in staged chunks: {staged_chunks}."
                    )
                break  # All channels have finished

        await ch_out.data.drain(ctx)


@define_py_node()
async def multicast_node_bounded(
    ctx: Context,
    ch_in: ChannelPair,
    *chs_out: ChannelPair,
) -> None:
    """
    Bounded multicast node for rapidsmpf.

    Each chunk is broadcasted to all output channels
    as it arrives.

    Parameters
    ----------
    ctx
        The context.
    ch_in
        The input ChannelPair.
    chs_out
        The output ChannelPairs.
    """
    # TODO: Use multiple streams
    async with shutdown_on_error(ctx, ch_in.data, *[ch.data for ch in chs_out]):
        while (msg := await ch_in.data.recv(ctx)) is not None:
            table_chunk = TableChunk.from_message(msg)
            for ch_out in chs_out:
                await ch_out.data.send(
                    ctx,
                    Message(
                        TableChunk.from_pylibcudf_table(
                            table_chunk.sequence_number,
                            table_chunk.table_view(),
                            table_chunk.stream,
                            exclusive_view=False,
                        )
                    ),
                )

        await asyncio.gather(*(ch.data.drain(ctx) for ch in chs_out))


@define_py_node()
async def multicast_node_unbounded(
    ctx: Context,
    ch_in: ChannelPair,
    *chs_out: ChannelPair,
) -> None:
    """
    Unbounded multicast node for rapidsmpf.

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
    ctx
        The context.
    ch_in
        The input ChannelPair.
    chs_out
        The output ChannelPairs.
    """
    async with shutdown_on_error(ctx, ch_in.data, *[ch.data for ch in chs_out]):
        # FIFO buffer for each output channel
        output_buffers: list[list[Message]] = [[] for _ in chs_out]

        # Track active send/drain tasks for each output
        active_tasks: dict[int, asyncio.Task] = {}

        # Track which outputs need to be drained (set when no more input)
        needs_drain: set[int] = set()

        # Receive task
        recv_task: asyncio.Task | None = asyncio.create_task(ch_in.data.recv(ctx))

        # Flag to indicate we should start a new receive (for backpressure)
        can_receive: bool = True

        async def send_one_from_buffer(idx: int) -> None:
            """Send one buffered message for output idx."""
            if output_buffers[idx]:
                msg = output_buffers[idx].pop(0)
                await chs_out[idx].data.send(ctx, msg)

        async def drain_output(idx: int) -> None:
            """Drain output channel idx."""
            await chs_out[idx].data.drain(ctx)

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
                        chunk = TableChunk.from_message(msg)
                        for buffer in output_buffers:
                            message = Message(
                                TableChunk.from_pylibcudf_table(
                                    chunk.sequence_number,
                                    chunk.table_view(),
                                    chunk.stream,
                                    exclusive_view=False,
                                )
                            )
                            buffer.append(message)

                        # Don't receive next chunk until at least one send completes
                        can_receive = False
                        recv_task = asyncio.create_task(ch_in.data.recv(ctx))
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

    # Check if we are broadcasting any children
    bcast_indices: list[int] = []
    if len(ir.children) > 1:
        counts = [rec.state["partition_info"][c].count for c in ir.children]
        bcast_indices = (
            []
            if all(c == 1 for c in counts)
            else [i for i, c in enumerate(counts) if c == 1]
        )

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager()

    if len(ir.children) == 1:
        # Single-channel default node
        nodes.append(
            default_node_single(
                rec.state["ctx"],
                ir,
                channels[ir].reserve_input_slot(),
                channels[ir.children[0]].reserve_output_slot(),
            )
        )
    else:
        # Multi-channel default node
        nodes.append(
            default_node_multi(
                rec.state["ctx"],
                ir,
                channels[ir].reserve_input_slot(),
                tuple(channels[c].reserve_output_slot() for c in ir.children),
                bcast_indices=bcast_indices,
            )
        )

    return nodes, channels


@define_py_node()
async def empty_node(
    ctx: Context,
    ir: Empty,
    ch_out: ChannelPair,
) -> None:
    """
    Empty node for rapidsmpf - produces a single empty chunk.

    Parameters
    ----------
    ctx
        The context.
    ir
        The Empty node.
    ch_out
        The output ChannelPair.
    """
    async with shutdown_on_error(ctx, ch_out.data):
        # Evaluate the IR node to create an empty DataFrame
        df: DataFrame = ir.do_evaluate(*ir._non_child_args)

        # Return the output chunk (empty but with correct schema)
        chunk = TableChunk.from_pylibcudf_table(
            0, df.table, DEFAULT_STREAM, exclusive_view=True
        )
        await ch_out.data.send(ctx, Message(chunk))

        await ch_out.data.drain(ctx)


@generate_ir_sub_network.register(Empty)
def _(ir: Empty, rec: SubNetGenerator) -> tuple[list[Any], dict[IR, ChannelManager]]:
    """Generate network for Empty node - produces one empty chunk."""
    ctx = rec.state["ctx"]
    channels: dict[IR, ChannelManager] = {ir: ChannelManager()}
    nodes: list[Any] = [empty_node(ctx, ir, channels[ir].reserve_input_slot())]
    return nodes, channels


def generate_ir_sub_network_wrapper(
    ir: IR, rec: SubNetGenerator
) -> tuple[list[Any], dict[IR, ChannelManager]]:
    """
    Generate a sub-network for the RapidsMPF streaming engine.

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

    # Check if this node needs multicast
    if (multicast_info := rec.state["multicast_nodes"].get(ir)) is not None:
        count = multicast_info.num_consumers
        manager = ChannelManager(count=count)
        if multicast_info.unbounded:
            nodes.append(
                multicast_node_unbounded(
                    rec.state["ctx"],
                    channels[ir].reserve_output_slot(),
                    *[manager.reserve_input_slot() for _ in range(count)],
                )
            )
        else:  # "bounded"
            nodes.append(
                multicast_node_bounded(
                    rec.state["ctx"],
                    channels[ir].reserve_output_slot(),
                    *[manager.reserve_input_slot() for _ in range(count)],
                )
            )
        channels[ir] = manager
    return nodes, channels
