# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Union logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.channel import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.ir import Union
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import define_py_node, shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    process_children,
)

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


@define_py_node()
async def union_node(
    ctx: Context,
    ir: Union,
    ch_out: ChannelPair,
    *chs_in: ChannelPair,
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
        The output ChannelPair.
    chs_in
        The input ChannelPairs.
    """
    # TODO: Use multiple streams
    async with shutdown_on_error(ctx, *[ch.data for ch in chs_in], ch_out.data):
        seq_num_offset = 0
        for ch_in in chs_in:
            num_ch_chunks = 0
            while (msg := await ch_in.data.recv(ctx)) is not None:
                table_chunk = TableChunk.from_message(msg)
                num_ch_chunks += 1
                await ch_out.data.send(
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

        await ch_out.data.drain(ctx)


@generate_ir_sub_network.register(Union)
def _(ir: Union, rec: SubNetGenerator) -> tuple[list[Any], dict[IR, ChannelManager]]:
    # Union operation.
    # Pass-through all child chunks in channel order.

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager()

    # Add simple python node
    nodes.append(
        union_node(
            rec.state["ctx"],
            ir,
            channels[ir].reserve_input_slot(),
            *[channels[c].reserve_output_slot() for c in ir.children],
        )
    )
    return nodes, channels
