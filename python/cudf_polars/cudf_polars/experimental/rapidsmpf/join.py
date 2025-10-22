# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.streaming.core.channel import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Join
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import (
    default_node_multi,
    define_py_node,
    shutdown_on_error,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    process_children,
)
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


async def get_small_table(
    ctx: Context,
    small_child: IR,
    ch_small: ChannelPair,
) -> list[DataFrame]:
    """
    Get the small-table DataFrame partitions from the small-table ChannelPair.

    Parameters
    ----------
    ctx
        The context.
    small_child
        The small-table child IR node.
    ch_small
        The small-table ChannelPair.

    Returns
    -------
    list[DataFrame]
        The small-table DataFrame partitions.
    """
    small_chunks = []
    while (msg := await ch_small.data.recv(ctx)) is not None:
        small_chunks.append(TableChunk.from_message(msg))

    if len(small_chunks) == 0:
        raise ValueError("Empty small side")

    return [
        DataFrame.from_table(
            small_chunk.table_view(),
            list(small_child.schema.keys()),
            list(small_child.schema.values()),
            small_chunk.stream,
        )
        for small_chunk in small_chunks
    ]


@define_py_node()
async def broadcast_join_node(
    ctx: Context,
    ir: Join,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    broadcast_side: Literal["left", "right"],
) -> None:
    """
    Join node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The Join IR node.
    ch_out
        The output ChannelPair.
    ch_left
        The left input ChannelPair.
    ch_right
        The right input ChannelPair.
    broadcast_side
        The side to broadcast.
    """
    async with shutdown_on_error(ctx, ch_left.data, ch_right.data, ch_out.data):
        if broadcast_side == "right":
            # Broadcast right, stream left
            small_ch = ch_right
            large_ch = ch_left
            small_child = ir.children[1]
            large_child = ir.children[0]
        else:
            # Broadcast left, stream right
            small_ch = ch_left
            large_ch = ch_right
            small_child = ir.children[0]
            large_child = ir.children[1]

        # Collect small-side chunks
        small_dfs = list(
            chain.from_iterable(
                await asyncio.gather(get_small_table(ctx, small_child, small_ch))
            )
        )
        if ir.options[0] != "Inner":
            # TODO: Use local repartitioning for non-inner joins
            small_dfs = [_concat(*small_dfs)]

        # Stream through large side, joining with the small-side
        while (msg := await large_ch.data.recv(ctx)) is not None:
            large_chunk = TableChunk.from_message(msg)
            large_df = DataFrame.from_table(
                large_chunk.table_view(),
                list(large_child.schema.keys()),
                list(large_child.schema.values()),
                large_chunk.stream,
            )

            # Perform the join
            results = [
                (
                    await asyncio.to_thread(
                        ir.do_evaluate,
                        *ir._non_child_args,
                        *(
                            [large_df, small_df]
                            if broadcast_side == "right"
                            else [small_df, large_df]
                        ),
                    )
                ).table
                for small_df in small_dfs
            ]

            # Send output chunk
            build_stream = DEFAULT_STREAM
            await ch_out.data.send(
                ctx,
                Message(
                    TableChunk.from_pylibcudf_table(
                        large_chunk.sequence_number,
                        (
                            results[0]
                            if len(results) == 1
                            else plc.concatenate.concatenate(results, build_stream)
                        ),
                        build_stream,
                        exclusive_view=True,
                    )
                ),
            )

        await ch_out.data.drain(ctx)


@generate_ir_sub_network.register(Join)
def _(ir: Join, rec: SubNetGenerator) -> tuple[list[Any], dict[IR, ChannelManager]]:
    # Join operation.
    left, right = ir.children
    partition_info = rec.state["partition_info"]
    output_count = partition_info[ir].count

    left_count = partition_info[left].count
    right_count = partition_info[right].count
    left_partitioned = (
        partition_info[left].partitioned_on == ir.left_on and left_count == output_count
    )
    right_partitioned = (
        partition_info[right].partitioned_on == ir.right_on
        and right_count == output_count
    )

    pwise_join = output_count == 1 or (left_partitioned and right_partitioned)

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager()

    if pwise_join:
        # Partition-wise join (use default_node_multi)
        nodes.append(
            default_node_multi(
                rec.state["ctx"],
                ir,
                channels[ir].reserve_input_slot(),
                (
                    channels[left].reserve_output_slot(),
                    channels[right].reserve_output_slot(),
                ),
                bcast_indices=[],
            )
        )
        return nodes, channels

    else:
        # Broadcast join (use broadcast_join_node)
        broadcast_side: Literal["left", "right"]
        if left_count >= right_count:
            # Broadcast right, stream left
            broadcast_side = "right"
        else:
            broadcast_side = "left"

        nodes.append(
            broadcast_join_node(
                rec.state["ctx"],
                ir,
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                broadcast_side=broadcast_side,
            )
        )
        return nodes, channels
