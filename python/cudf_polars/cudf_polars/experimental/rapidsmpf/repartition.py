# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Re-chunking logic for the RapidsMPF streaming engine."""

from __future__ import annotations

import math
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
    ir: Repartition,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    *,
    max_chunks: int | None,
) -> None:
    """
    Concatenate node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The Repartition IR node.
    ch_out
        The output channel.
    ch_in
        The input channel.
    max_chunks
        The maximum number of chunks to concatenate at once.
        If `None`, concatenate all input chunks.
    """
    # TODO: Use multiple streams
    max_chunks = max(2, max_chunks) if max_chunks else None
    async with shutdown_on_error(ctx, ch_in, ch_out):
        build_stream = DEFAULT_STREAM

        seq_num = 0
        while True:
            chunks: list[TableChunk] = []
            msg: TableChunk | None = None

            # Collect chunks up to max_chunks or until end of stream
            while len(chunks) < (max_chunks or float("inf")):
                msg = await ch_in.recv(ctx)
                if msg is None:
                    break
                chunk = TableChunk.from_message(msg)
                chunks.append(chunk)

            # Process collected chunks
            if chunks:
                table = (
                    chunks[0].table_view()
                    if len(chunks) == 1
                    else plc.concatenate.concatenate(
                        [chunk.table_view() for chunk in chunks], build_stream
                    )
                )
                await ch_out.send(
                    ctx,
                    Message(
                        TableChunk.from_pylibcudf_table(seq_num, table, build_stream)
                    ),
                )
                seq_num += 1

            # Break if we reached end of stream
            if msg is None:
                break

        await ch_out.drain(ctx)


@generate_ir_sub_network.register(Repartition)
def _(
    ir: Repartition, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, list[Any]]]:
    # Repartition node.

    partition_info = rec.state["partition_info"]
    max_chunks: int | None = None
    if partition_info[ir].count > 1:
        count_output = partition_info[ir].count
        count_input = partition_info[ir.children[0]].count
        if count_input < count_output:
            raise ValueError("Repartitioning to more chunks is not supported.")
        # Make sure max_chunks is at least 2
        max_chunks = max(2, math.ceil(count_input / count_output))

    # Process children
    nodes, channels = rec(ir.children[0])

    # Create output channel
    channels[ir] = [Channel()]

    # Add python node
    nodes[ir] = [
        concatenate_node(
            rec.state["ctx"],
            ir,
            channels[ir][0],
            channels[ir.children[0]].pop(),
            max_chunks=max_chunks,
        )
    ]
    return nodes, channels
