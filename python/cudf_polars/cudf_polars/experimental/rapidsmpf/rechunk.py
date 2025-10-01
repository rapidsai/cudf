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
    goal
        The goal of the local rechunking operation.
        The default is 'single', which concatenates all
        input data into a single chunk.
    df
        The input IR node.
    """

    __slots__ = ("goal",)
    _non_child = ("schema", "goal")

    def __init__(
        self,
        schema: Schema,
        goal: Literal["single"],
        df: IR,
    ):
        self.schema = schema
        self._non_child_args = ()
        self.children = (df,)
        self.goal = goal
        if goal != "single":  # pragma: no cover
            # TODO: Add other rechunking goals, e.g. 'row_count', 'byte_size'
            raise NotImplementedError("Only 'single' rechunking is supported for now.")


@define_py_node()
async def rechunk_node(
    ctx: Context,
    ir: Rechunk,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
) -> None:
    """
    Rechunk node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The IR node.
    ch_in
        The input channel.
    ch_out
        The output channel.
    """
    # TODO: Use multiple streams
    # TODO: Shupport single_chunk=False
    if ir.goal != "single":  # pragma: no cover
        raise NotImplementedError("Only single rechunking is supported for now.")

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

    # Process children
    nodes, channels = rec(ir.children[0])

    # Create output channel
    channels[ir] = Channel()

    # Add python node
    nodes[ir] = [
        rechunk_node(
            rec.state["ctx"],
            ir=ir,
            ch_in=channels[ir.children[0]],
            ch_out=channels[ir],
        )
    ]
    return nodes, channels
