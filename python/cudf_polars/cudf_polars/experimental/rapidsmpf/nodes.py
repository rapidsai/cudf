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
from cudf_polars.dsl.ir import IR, Empty
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
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
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
        The output channel.
    ch_in
        The input channel.

    Notes
    -----
    Chunks are processed in the order they are received.
    """
    async with shutdown_on_error(ctx, ch_in, ch_out):
        while (msg := await ch_in.recv(ctx)) is not None:
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
            await ch_out.send(ctx, Message(chunk))

        await ch_out.drain(ctx)


@define_py_node()
async def default_node_multi(
    ctx: Context,
    ir: IR,
    ch_out: Channel[TableChunk],
    *chs_in: Channel[TableChunk],
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
        The output channel.
    chs_in
        The input channels.
    bcast_indices
        The indices of the broadcasted children.

    Notes
    -----
    Input chunks are aligned for evaluation.
    """
    # TODO: Use multiple streams
    async with shutdown_on_error(ctx, *chs_in, ch_out):
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
                        msg := await ch_in.recv(ctx)
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
                await ch_out.send(
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

        # Drain the output channel
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
        while (msg := await ch_in.recv(ctx)) is not None:
            table_chunk = TableChunk.from_message(msg)
            for ch_out in chs_out:
                await ch_out.send(
                    ctx,
                    Message(
                        TableChunk.from_pylibcudf_table(
                            table_chunk.sequence_number,
                            table_chunk.table_view(),
                            table_chunk.stream,
                            # NOTE: Should we just copy the table chunk?
                            exclusive_view=False,
                        )
                    ),
                )
        await asyncio.gather(*(ch.drain(ctx) for ch in chs_out))


@define_py_node()
async def passthrough_node(
    ctx: Context,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    *,
    union_dependency: bool,
) -> None:
    """
    Passthrough node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    union_dependency
        Whether a Union node depends on this passthrough node.
        If so, we must forward all chunks immediately
        to avoid blocking progress in a multicast node.
    """
    async with shutdown_on_error(ctx, ch_in, ch_out):
        # Unfortunately, we must forward all chunks immediately.
        # Otherwise, we may block progress in a multicast node.
        sends = []
        while (msg := await ch_in.recv(ctx)) is not None:
            if union_dependency:
                sends.append(asyncio.create_task(ch_out.send(ctx, msg)))
            else:
                await ch_out.send(ctx, msg)
        if sends:
            await asyncio.gather(*sends)
        await ch_out.drain(ctx)


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

    if len(ir.children) == 1:
        # Single-channel default node
        nodes[ir] = [
            default_node_single(
                rec.state["ctx"],
                ir,
                channels[ir][0],
                channels[ir.children[0]].pop(),
            )
        ]
    else:
        # Multi-channel default node
        counts = [rec.state["partition_info"][c].count for c in ir.children]
        bcast_indices = (
            []
            if all(c == 1 for c in counts)
            else [i for i, c in enumerate(counts) if c == 1]
        )
        nodes[ir] = [
            default_node_multi(
                rec.state["ctx"],
                ir,
                channels[ir][0],
                *[channels[c].pop() for c in ir.children],
                bcast_indices=bcast_indices,
            )
        ]

    return nodes, channels


@define_py_node()
async def empty_node(
    ctx: Context,
    ir: Empty,
    ch_out: Channel[TableChunk],
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
        The output channel.
    """
    async with shutdown_on_error(ctx, ch_out):
        # Evaluate the IR node to create an empty DataFrame
        df: DataFrame = ir.do_evaluate(*ir._non_child_args)

        # Return the output chunk (empty but with correct schema)
        chunk = TableChunk.from_pylibcudf_table(
            0, df.table, DEFAULT_STREAM, exclusive_view=True
        )
        await ch_out.send(ctx, Message(chunk))

        await ch_out.drain(ctx)


@generate_ir_sub_network.register(Empty)
def _(ir: Empty, rec: SubNetGenerator) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    """Generate network for Empty node - produces one empty chunk."""
    ctx = rec.state["ctx"]
    ch_out = Channel()
    nodes: dict[IR, list[Any]] = {ir: [empty_node(ctx, ir, ch_out)]}
    channels: dict[IR, list[Any]] = {ir: [ch_out]}
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
        inter_chs = [Channel() for _ in range(count)]
        output_chs = [Channel() for _ in range(count)]
        nodes[ir].append(multicast_node(rec.state["ctx"], channels[ir][0], *inter_chs))
        for ch_in, ch_out in zip(inter_chs, output_chs, strict=True):
            # We need a passthrough node to ensure that downstram
            # consumers can recv chunks at different rates
            nodes[ir].append(
                passthrough_node(
                    rec.state["ctx"],
                    ch_out,
                    ch_in,
                    union_dependency=rec.state["union_dependency"],
                )
            )
        channels[ir] = output_chs
    return nodes, channels
