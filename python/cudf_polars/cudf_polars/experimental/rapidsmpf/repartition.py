# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Re-chunking logic for the RapidsMPF streaming engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.repartition import Repartition

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


@define_py_node()
async def concatenate_node(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
) -> None:
    """
    Concatenate node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ch_in
        The input channel.
    ch_out
        The output channel.
    """
    # TODO: Use multiple streams
    async with shutdown_on_error(ctx, ch_in, ch_out):
        chunks = []
        build_stream = DEFAULT_STREAM
        while (msg := await ch_in.recv(ctx)) is not None:
            chunk = TableChunk.from_message(msg)
            chunks.append(chunk)

        table = (
            chunks[0].table_view()
            if len(chunks) == 1
            else plc.concatenate.concatenate(
                [chunk.table_view() for chunk in chunks], build_stream
            )
        )
        await ch_out.send(
            ctx, Message(TableChunk.from_pylibcudf_table(0, table, build_stream))
        )
        await ch_out.drain(ctx)


@generate_ir_sub_network.register(Repartition)
def _(
    ir: Repartition, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    # Repartition node.

    # TODO: Support other repartitioning strategies
    partition_info = rec.state["partition_info"]
    assert partition_info[ir].count == 1, "Only concatenation is supported for now."

    # Process children
    nodes, channels = rec(ir.children[0])

    # Create output channel
    channels[ir] = Channel()

    # Add python node
    nodes[ir] = [
        concatenate_node(
            rec.state["ctx"],
            ch_in=channels[ir.children[0]],
            ch_out=channels[ir],
        )
    ]
    return nodes, channels
