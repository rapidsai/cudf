# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Re-chunking logic for the RapidsMPF streaming engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.dsl.ir import IR
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.typing import Schema


class Rechunk(IR):
    """
    Rechunk input data locally.

    Parameters
    ----------
    schema
        The schema of the output data.
    strategy
        The strategy of the local rechunking operation.
        Only 'chunk_count' is supported for now.
    value
        The numerical value used by the local rechunking strategy.
        Only `1` is supported for now.
    df
        The input IR node.
    """

    __slots__ = ("strategy", "value")
    _non_child = ("schema", "strategy", "value")

    def __init__(
        self,
        schema: Schema,
        strategy: Literal["chunk_count", "row_count", "byte_count"],
        value: int,
        df: IR,
    ):
        self.schema = schema
        self._non_child_args = ()
        self.children = (df,)
        self.strategy = strategy
        self.value = value
        if strategy != "chunk_count" or value != 1:  # pragma: no cover
            # TODO: Support other rechunking strategies
            raise NotImplementedError("Only rechunking to 1 is supported for now.")


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


@generate_ir_sub_network.register(Rechunk)
def _(ir: Rechunk, rec: SubNetGenerator) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    # Rechunk node.

    # TODO: Support other rechunking strategies
    if ir.strategy != "chunk_count" or ir.value != 1:  # pragma: no cover
        raise NotImplementedError("Only rechunking to 1 chunk is supported for now.")

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
