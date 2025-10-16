# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Union logic for the RapidsMPF streaming engine."""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.ir import Union
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import define_py_node, shutdown_on_error

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


@define_py_node()
async def union_node(
    ctx: Context,
    ir: Union,
    ch_out: Channel[TableChunk],
    *chs_in: Channel[TableChunk],
) -> None:
    """
    Union node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The Union IR node.
    ch_out
        The output channel.
    chs_in
        The input channels.
    """
    # TODO: Use multiple streams
    async with shutdown_on_error(ctx, *chs_in, ch_out):
        seq_num_offset = 0
        for ch_in in chs_in:
            num_ch_chunks = 0
            while (msg := await ch_in.recv(ctx)) is not None:
                table_chunk = TableChunk.from_message(msg)
                num_ch_chunks += 1
                await ch_out.send(
                    ctx,
                    Message(
                        TableChunk.from_pylibcudf_table(
                            table_chunk.sequence_number + seq_num_offset,
                            table_chunk.table_view(),
                            table_chunk.stream,
                            exclusive_view=True,
                        )
                    ),
                )
            seq_num_offset += num_ch_chunks

        await ch_out.drain(ctx)


@generate_ir_sub_network.register(Union)
def _(
    ir: Union, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, list[Any]]]:
    # Union operation.
    # Pass-through all child chunks in channel order.

    # Process children
    rec.state["union_dependency"] = True
    _nodes, _channels = zip(*(rec(c) for c in ir.children), strict=True)
    nodes = reduce(operator.or_, _nodes)
    channels = reduce(operator.or_, _channels)

    # Create output channel
    channels[ir] = [Channel()]

    # Add simple python node
    nodes[ir] = [
        union_node(
            rec.state["ctx"],
            ir,
            channels[ir][0],
            *[channels[c].pop() for c in ir.children],
        )
    ]
    return nodes, channels
