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
async def pwise_node(
    ctx: Context,
    ir: IR,
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
    ch_out
        The output channel.
    chs_in
        The input channels.
    """
    # TODO: Use multiple streams
    n_children = len(chs_in)
    async with shutdown_on_error(ctx, *chs_in, ch_out):
        seq_num = 0
        while True:
            input_dfs: list[DataFrame] = []
            for i, ch_in in enumerate(chs_in):
                if (msg := await ch_in.recv(ctx)) is not None:
                    input_dfs.append(
                        DataFrame.from_table(
                            TableChunk.from_message(msg).table_view(),
                            list(ir.children[i].schema.keys()),
                            list(ir.children[i].schema.values()),
                        )
                    )

            if input_dfs:
                if len(input_dfs) != n_children:
                    raise ValueError(
                        "Number of input chunks does not match the child count."
                    )

                # Evaluate the IR node
                df: DataFrame = ir.do_evaluate(*ir._non_child_args, *input_dfs)

                # Return the output chunk
                chunk = TableChunk.from_pylibcudf_table(
                    seq_num, df.table, DEFAULT_STREAM
                )
                await ch_out.send(ctx, Message(chunk))
                seq_num += 1
            else:
                break
        await ch_out.drain(ctx)


@generate_ir_sub_network.register(IR)
def _(ir: IR, rec: SubNetGenerator) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    # Default generate_ir_sub_network logic.
    # Use simple pointwise node.

    # Process children
    _nodes, _channels = zip(*(rec(c) for c in ir.children), strict=True)
    nodes = reduce(operator.or_, _nodes)
    channels = reduce(operator.or_, _channels)

    # Create output channel
    channels[ir] = Channel()

    # Add simple python node
    nodes[ir] = [
        pwise_node(
            rec.state["ctx"],
            ir,
            channels[ir],
            *[channels[c] for c in ir.children],
        )
    ]
    return nodes, channels
