# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core node definitions for the RapidsMPF streaming engine."""

from __future__ import annotations

import asyncio
import operator
from contextlib import asynccontextmanager
from functools import reduce
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


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
    try:
        yield
    except Exception:
        await asyncio.gather(*(ch.shutdown(ctx) for ch in channels))
        raise


@define_py_node()
async def default_node(
    ctx: Context,
    ir: IR,
    bcast_indices: list[int],
    ch_out: Channel[TableChunk],
    *chs_in: Channel[TableChunk],
) -> None:
    """
    Pointwise node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The IR node.
    bcast_indices: list[int],
        The indices of the broadcasted children.
    ch_out
        The output channel.
    chs_in
        The input channels.
    """
    # TODO: Use multiple streams
    async with shutdown_on_error(ctx, *chs_in, ch_out):
        seq_num = 0
        bcast_data = {}

        # First, collect broadcast data (if any)
        for i in bcast_indices:
            msg = await chs_in[i].recv(ctx)
            if msg is not None:
                chunk = DataFrame.from_table(
                    TableChunk.from_message(msg).table_view(),
                    list(ir.children[i].schema.keys()),
                    list(ir.children[i].schema.values()),
                )
                bcast_data[i] = chunk

        # Then, process streaming data from non-broadcast channels
        output_chunks = []
        while True:
            chunks = {}
            has_data = False

            for i, ch_in in enumerate(chs_in):
                if i not in bcast_indices:
                    msg = await ch_in.recv(ctx)
                    if msg is not None:
                        chunk = DataFrame.from_table(
                            TableChunk.from_message(msg).table_view(),
                            list(ir.children[i].schema.keys()),
                            list(ir.children[i].schema.values()),
                        )
                        chunks[i] = chunk
                        has_data = True

            if not has_data:
                break

            # Evaluate the IR node
            child_data = [chunks.get(i, bcast_data.get(i)) for i in range(len(chs_in))]
            df: DataFrame = ir.do_evaluate(*ir._non_child_args, *child_data)

            # Store output chunk instead of sending immediately
            chunk = TableChunk.from_pylibcudf_table(seq_num, df.table, DEFAULT_STREAM)
            output_chunks.append(chunk)
            seq_num += 1

        # Send all output chunks at once to avoid circular blocking
        for i, chunk in enumerate(output_chunks):
            await ch_out.send(ctx, Message(chunk))

        # Make sure input bcast channels are empty
        for i in bcast_indices:
            remaining = await chs_in[i].recv(ctx)
            assert remaining is None, f"Broadcast channel {i} should be empty"

        # Drain the output channel
        await ch_out.drain(ctx)


async def forward_to_channel(
    ctx: Context,
    chunks: list[plc.Table],
    ch_out: Channel[TableChunk],
    channel_id: int = 0,
) -> None:
    """
    Send all chunks to a single output channel, then drain it.

    This ensures atomic processing: all sends to this channel complete
    before the channel is drained.

    Parameters
    ----------
    ctx
        The context.
    chunks
        The chunks to send.
    ch_out
        The output channel.
    channel_id
        Identifier for debugging.
    """
    for seq_num, chunk in enumerate(chunks):
        await ch_out.send(
            ctx,
            Message(TableChunk.from_pylibcudf_table(seq_num, chunk, DEFAULT_STREAM)),
        )
    await ch_out.drain(ctx)


@define_py_node()
async def multicast_node(
    ctx: Context,
    ch_in: Channel[TableChunk],
    *chs_out: Channel[TableChunk],
) -> None:
    """
    Multicast node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ch_in
        The input channel.
    chs_out
        The output channels.
    """
    # TODO: Use multiple streams
    async with shutdown_on_error(ctx, ch_in, *chs_out):
        # Collect all chunks from input channel
        chunks: list[TableChunk] = []
        while (msg := await ch_in.recv(ctx)) is not None:
            chunks.append(TableChunk.from_message(msg).table_view())

        # Send chunks to all output channels using atomic per-channel processing
        # This ensures that channels consuming at different rates don't block each other
        # (e.g., streaming operations vs fallback repartition operations)
        await asyncio.gather(
            *(
                forward_to_channel(ctx, chunks, ch_out, channel_id=i)
                for i, ch_out in enumerate(chs_out)
            )
        )


@generate_ir_sub_network.register(IR)
def _(ir: IR, rec: SubNetGenerator) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    # Default generate_ir_sub_network logic.
    # Use simple pointwise node.

    # Process children
    nodes: dict[IR, list[Any]] = {}
    channels: dict[IR, list[Any]] = {}
    if ir.children:
        _nodes, _channels = zip(*(rec(c) for c in ir.children), strict=True)
        nodes = reduce(operator.or_, _nodes)
        channels = reduce(operator.or_, _channels)

    # Create output channel
    channels[ir] = [Channel()]

    # Add simple python node
    partition_info = rec.state["partition_info"]
    # TODO: What about multiple broadcasted partitions?
    # We are tracking broadcasted partitions in PartitionInfo,
    # but this logic only handles the single-partition case.
    counts = [partition_info[c].count for c in ir.children]
    bcast_indices = (
        []
        if all(c == 1 for c in counts)
        else [i for i, c in enumerate(counts) if c == 1]
    )
    nodes[ir] = [
        default_node(
            rec.state["ctx"],
            ir,
            bcast_indices,
            channels[ir][0],
            *[channels[c].pop() for c in ir.children],
        )
    ]
    return nodes, channels


def generate_ir_sub_network_wrapper(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, list[Any]]]:
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
        Dictionary mapping between each IR node and its
        corresponding streaming-network node(s).
    channels
        Dictionary mapping between each IR node and its
        corresponding streaming-network output channels.
    """
    nodes, channels = generate_ir_sub_network(ir, rec)
    if (count := rec.state["output_ch_count"][ir]) > 1:
        output_chs = [Channel() for _ in range(count)]
        nodes[ir].append(multicast_node(rec.state["ctx"], *channels[ir], *output_chs))
        channels[ir] = output_chs
    return nodes, channels
