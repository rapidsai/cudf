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
    broadcasted: list[int],
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
    broadcasted: list[int],
        The indices of the broadcasted children.
    ch_out
        The output channel.
    chs_in
        The input channels.
    """
    # TODO: Use multiple streams
    n_children = len(chs_in)
    async with shutdown_on_error(ctx, *chs_in, ch_out):
        # Collect broadcasted partitions first
        broadcasted_data = {}
        for i in range(len(broadcasted)):
            if (msg := await chs_in[i].recv(ctx)) is not None:
                broadcasted_data[i] = DataFrame.from_table(
                    TableChunk.from_message(msg).table_view(),
                    list(ir.children[i].schema.keys()),
                    list(ir.children[i].schema.values()),
                )

        seq_num = 0
        while True:
            input_dfs: list[DataFrame] = []
            for i, ch_in in enumerate(chs_in):
                if (_df := broadcasted_data.get(i)) is not None:
                    input_dfs.append(_df)
                elif (msg := await ch_in.recv(ctx)) is not None:
                    input_dfs.append(
                        DataFrame.from_table(
                            TableChunk.from_message(msg).table_view(),
                            list(ir.children[i].schema.keys()),
                            list(ir.children[i].schema.values()),
                        )
                    )

            if len(input_dfs) == n_children:
                # Evaluate the IR node
                df: DataFrame = ir.do_evaluate(*ir._non_child_args, *input_dfs)

                # Return the output chunk
                chunk = TableChunk.from_pylibcudf_table(
                    seq_num, df.table, DEFAULT_STREAM
                )
                await ch_out.send(ctx, Message(chunk))
                seq_num += 1
            elif len(broadcasted) == len(input_dfs):
                break  # All done
            else:
                raise ValueError(
                    "Number of input chunks does not match the child count."
                )
            if len(broadcasted) == n_children:
                break  # All done
        await ch_out.drain(ctx)


@define_py_node()
async def multicast_node(
    ctx: Context,
    ch_in: Channel[TableChunk],
    *chs_out: Channel[TableChunk],
) -> None:
    """
    Fan-out node for rapidsmpf.

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
        seq_num = 0
        while (msg := await ch_in.recv(ctx)) is not None:
            chunks.append(TableChunk.from_message(msg))
            print(f"Received chunk {seq_num} from input channel", flush=True)
            seq_num += 1

        # Send chunks to all output channels
        for i, ch_out in enumerate(chs_out):
            print(f"Sending chunks to output channel {i}", flush=True)
            for seq_num, chunk in enumerate(chunks):
                print(f"Sending chunk {seq_num} to output channel {i}", flush=True)
                await ch_out.send(
                    ctx,
                    Message(
                        TableChunk.from_pylibcudf_table(
                            seq_num,
                            chunk.table_view(),
                            DEFAULT_STREAM,
                        )
                    ),
                )

        for ch_out in chs_out:
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

    # Add simple python node
    partition_info = rec.state["partition_info"]
    # TODO: What about multiple broadcasted partitions?
    # We are tracking broadcasted partitions in PartitionInfo,
    # but this logic only handles the single-partition case.
    counts = [partition_info[c].count for c in ir.children]
    broadcasted = (
        []
        if all(count == 1 for count in counts)
        else [i for i, c in enumerate(ir.children) if partition_info[c].count == 1]
    )
    nodes[ir] = [
        default_node(
            rec.state["ctx"],
            ir,
            broadcasted,
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
