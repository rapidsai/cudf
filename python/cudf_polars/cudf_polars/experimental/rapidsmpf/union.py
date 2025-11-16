# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Union logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.message import Message
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

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


@define_py_node()
async def union_node(
    context: Context,
    ir: Union,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    *chs_in: ChannelPair,
) -> None:
    """
    Union node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Union IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    chs_in
        The input ChannelPairs.
    """
    # TODO: Use multiple streams
    async with shutdown_on_error(context, *[ch.data for ch in chs_in], ch_out.data):
        seq_num_offset = 0
        for ch_in in chs_in:
            num_ch_chunks = 0
            while (msg := await ch_in.data.recv(context)) is not None:
                num_ch_chunks += 1
                await ch_out.data.send(
                    context,
                    Message(
                        msg.sequence_number + seq_num_offset,
                        TableChunk.from_message(msg).make_available_and_spill(
                            context.br(), allow_overbooking=True
                        ),
                    ),
                )
            seq_num_offset += num_ch_chunks

        await ch_out.data.drain(context)


@generate_ir_sub_network.register(Union)
def _(ir: Union, rec: SubNetGenerator) -> tuple[list[Any], dict[IR, ChannelManager]]:
    # Union operation.
    # Pass-through all child chunks in channel order.

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Add simple python node
    nodes.append(
        union_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            *[channels[c].reserve_output_slot() for c in ir.children],
        )
    )
    return nodes, channels
