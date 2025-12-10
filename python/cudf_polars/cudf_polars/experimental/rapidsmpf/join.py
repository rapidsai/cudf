# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Join
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
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
    Metadata,
    process_children,
)
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


async def get_small_table(
    context: Context,
    small_child: IR,
    ch_small: ChannelPair,
) -> tuple[list[DataFrame], int]:
    """
    Get the small-table DataFrame partitions from the small-table ChannelPair.

    Parameters
    ----------
    context
        The rapidsmpf context.
    small_child
        The small-table child IR node.
    ch_small
        The small-table ChannelPair.

    Returns
    -------
    The small-table DataFrame partitions and the size of the small-side in bytes.
    """
    small_chunks = []
    small_size = 0
    while (msg := await ch_small.data.recv(context)) is not None:
        small_chunks.append(
            TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
        )
        small_size += small_chunks[-1].data_alloc_size(MemoryType.DEVICE)
    assert small_chunks, "Empty small side"

    return [
        DataFrame.from_table(
            small_chunk.table_view(),
            list(small_child.schema.keys()),
            list(small_child.schema.values()),
            small_chunk.stream,
        )
        for small_chunk in small_chunks
    ], small_size


@define_py_node()
async def broadcast_join_node(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    broadcast_side: Literal["left", "right"],
    collective_id: int,
    target_partition_size: int,
) -> None:
    """
    Join node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Join IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    ch_left
        The left input ChannelPair.
    ch_right
        The right input ChannelPair.
    broadcast_side
        The side to broadcast.
    collective_id
        Pre-allocated collective ID for this operation.
    target_partition_size
        The target partition size in bytes.
    """
    async with shutdown_on_error(
        context,
        ch_left.metadata,
        ch_left.data,
        ch_right.metadata,
        ch_right.data,
        ch_out.metadata,
        ch_out.data,
    ):
        # Receive metadata.
        left_metadata, right_metadata = await asyncio.gather(
            ch_left.recv_metadata(context),
            ch_right.recv_metadata(context),
        )

        partitioned_on: tuple[str, ...] = ()
        if broadcast_side == "right":
            # Broadcast right, stream left
            small_ch = ch_right
            large_ch = ch_left
            small_child = ir.children[1]
            large_child = ir.children[0]
            chunk_count = left_metadata.count
            partitioned_on = left_metadata.partitioned_on
            small_duplicated = right_metadata.duplicated
        else:
            # Broadcast left, stream right
            small_ch = ch_left
            large_ch = ch_right
            small_child = ir.children[0]
            large_child = ir.children[1]
            chunk_count = right_metadata.count
            small_duplicated = left_metadata.duplicated
            if ir.options[0] == "Right":
                partitioned_on = right_metadata.partitioned_on

        # Send metadata.
        output_metadata = Metadata(
            chunk_count,
            partitioned_on=partitioned_on,
            duplicated=left_metadata.duplicated and right_metadata.duplicated,
        )
        await ch_out.send_metadata(context, output_metadata)

        # Collect small-side
        small_dfs, small_size = await get_small_table(context, small_child, small_ch)
        need_allgather = context.comm().nranks > 1 and not small_duplicated
        if (
            ir.options[0] != "Inner" or small_size < target_partition_size
        ) and not need_allgather:
            # Pre-concat for non-inner joins, otherwise
            # we need a local shuffle, and face additional
            # memory pressure anyway.
            small_dfs = [_concat(*small_dfs, context=ir_context)]
        if need_allgather:
            allgather = AllGatherManager(context, collective_id)
            for s_id, small_df in enumerate(small_dfs):
                allgather.insert(
                    s_id,
                    TableChunk.from_pylibcudf_table(
                        small_df.table, small_df.stream, exclusive_view=True
                    ),
                )
            allgather.insert_finished()
            small_dfs = [
                DataFrame.from_table(
                    await allgather.extract_concatenated(small_df.stream),
                    list(small_child.schema.keys()),
                    list(small_child.schema.values()),
                    small_df.stream,
                )
            ]

        # Stream through large side, joining with the small-side
        while (msg := await large_ch.data.recv(context)) is not None:
            large_chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            seq_num = msg.sequence_number
            large_df = DataFrame.from_table(
                large_chunk.table_view(),
                list(large_child.schema.keys()),
                list(large_child.schema.values()),
                large_chunk.stream,
            )

            # Perform the join
            df = _concat(
                *[
                    (
                        await asyncio.to_thread(
                            ir.do_evaluate,
                            *ir._non_child_args,
                            *(
                                [large_df, small_df]
                                if broadcast_side == "right"
                                else [small_df, large_df]
                            ),
                            context=ir_context,
                        )
                    )
                    for small_df in small_dfs
                ],
                context=ir_context,
            )

            # Send output chunk
            await ch_out.data.send(
                context,
                Message(
                    seq_num,
                    TableChunk.from_pylibcudf_table(
                        df.table, df.stream, exclusive_view=True
                    ),
                ),
            )

        await ch_out.data.drain(context)


@generate_ir_sub_network.register(Join)
def _(
    ir: Join, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
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
    channels[ir] = ChannelManager(rec.state["context"])

    if pwise_join:
        # Partition-wise join (use default_node_multi)
        partitioning_index = 1 if ir.options[0] == "Right" else 0
        nodes[ir] = [
            default_node_multi(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                (
                    channels[left].reserve_output_slot(),
                    channels[right].reserve_output_slot(),
                ),
                partitioning_index=partitioning_index,
            )
        ]
        return nodes, channels

    else:
        # Broadcast join (use broadcast_join_node)
        broadcast_side: Literal["left", "right"]
        if left_count >= right_count:
            # Broadcast right, stream left
            broadcast_side = "right"
        else:
            broadcast_side = "left"

        # Get target partition size
        config_options = rec.state["config_options"]
        executor = config_options.executor
        assert executor.name == "streaming", "Join node requires streaming executor"
        target_partition_size = executor.target_partition_size

        nodes[ir] = [
            broadcast_join_node(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                broadcast_side=broadcast_side,
                collective_id=rec.state["collective_id_map"][ir],
                target_partition_size=target_partition_size,
            )
        ]
        return nodes, channels
