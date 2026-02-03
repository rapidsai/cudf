# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
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
    chunk_to_frame,
    empty_table_chunk,
    opaque_reservation,
    process_children,
    recv_metadata,
    send_metadata,
)
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.channel_metadata import Partitioning

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


@define_py_node()
async def broadcast_join_node(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
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
        The output Channel[TableChunk].
    ch_left
        The left input Channel[TableChunk].
    ch_right
        The right input Channel[TableChunk].
    broadcast_side
        The side to broadcast.
    collective_id
        Pre-allocated collective ID for this operation.
    target_partition_size
        The target partition size in bytes.
    """
    async with shutdown_on_error(context, ch_left, ch_right, ch_out):
        # Receive metadata.
        left_metadata, right_metadata = await asyncio.gather(
            recv_metadata(ch_left, context),
            recv_metadata(ch_right, context),
        )

        partitioning: Partitioning | None = None
        if broadcast_side == "right":
            # Broadcast right, stream left
            small_ch = ch_right
            large_ch = ch_left
            small_child = ir.children[1]
            large_child = ir.children[0]
            # Preserve left-side partitioning metadata
            local_count = left_metadata.local_count
            partitioning = left_metadata.partitioning
            # Check if the right-side is already broadcasted
            small_duplicated = right_metadata.duplicated
        else:
            # Broadcast left, stream right
            small_ch = ch_left
            large_ch = ch_right
            small_child = ir.children[0]
            large_child = ir.children[1]
            # Preserve right-side partitioning metadata
            local_count = right_metadata.local_count
            if ir.options[0] == "Right":
                partitioning = right_metadata.partitioning
            # Check if the right-side is already broadcasted
            small_duplicated = left_metadata.duplicated

        # Send metadata.
        output_metadata = ChannelMetadata(
            local_count=local_count,
            partitioning=partitioning,
            # The result is only "duplicated" if both sides are duplicated
            duplicated=left_metadata.duplicated and right_metadata.duplicated,
        )
        await send_metadata(ch_out, context, output_metadata)

        # Collect small-side (may be empty if no data received)
        small_chunks: list[TableChunk] = []
        small_size = 0
        while (msg := await small_ch.recv(context)) is not None:
            small_chunks.append(
                TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
            )
            del msg
            small_size += small_chunks[-1].data_alloc_size(MemoryType.DEVICE)

        # Allgather is a collective - all ranks must participate even with no local data
        need_allgather = context.comm().nranks > 1 and not small_duplicated
        if need_allgather:
            allgather = AllGatherManager(context, collective_id)
            for s_id in range(len(small_chunks)):
                allgather.insert(s_id, small_chunks.pop(0))
            allgather.insert_finished()
            stream = ir_context.get_cuda_stream()
            # extract_concatenated returns a plc.Table, not a TableChunk
            small_dfs = [
                DataFrame.from_table(
                    await allgather.extract_concatenated(stream),
                    list(small_child.schema.keys()),
                    list(small_child.schema.values()),
                    stream,
                )
            ]
        elif len(small_chunks) > 1 and (
            ir.options[0] != "Inner" or small_size < target_partition_size
        ):
            # Pre-concat for non-inner joins, otherwise
            # we need a local shuffle, and face additional
            # memory pressure anyway.
            small_dfs = [
                _concat(
                    *[chunk_to_frame(chunk, small_child) for chunk in small_chunks],
                    context=ir_context,
                )
            ]
            small_chunks.clear()  # small_dfs is not a view of small_chunks anymore
        else:
            small_dfs = [
                chunk_to_frame(small_chunk, small_child) for small_chunk in small_chunks
            ]

        # Stream through large side, joining with the small-side
        seq_num = 0
        large_chunk_processed = False
        receiving_large_chunks = True
        while receiving_large_chunks:
            msg = await large_ch.recv(context)
            if msg is None:
                receiving_large_chunks = False
                if large_chunk_processed:
                    # Normal exit - We've processed all large-table data
                    break
                elif small_dfs:
                    # We received small-table data, but no large-table data.
                    # This may never happen, but we can handle it by generating
                    # an empty large-table chunk
                    stream = ir_context.get_cuda_stream()
                    large_chunk = empty_table_chunk(large_child, context, stream)
                else:
                    # We received no data for either the small or large table.
                    # Drain the output channel and return
                    await ch_out.drain(context)
                    return
            else:
                large_chunk_processed = True
                large_chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
                seq_num = msg.sequence_number
                del msg

            large_df = DataFrame.from_table(
                large_chunk.table_view(),
                list(large_child.schema.keys()),
                list(large_child.schema.values()),
                large_chunk.stream,
            )

            # Lazily create empty small table if small_dfs is empty
            if not small_dfs:
                stream = ir_context.get_cuda_stream()
                empty_small_chunk = empty_table_chunk(small_child, context, stream)
                small_dfs = [chunk_to_frame(empty_small_chunk, small_child)]

            large_chunk_size = large_chunk.data_alloc_size(MemoryType.DEVICE)
            input_bytes = large_chunk_size + small_size
            with opaque_reservation(context, input_bytes):
                df = _concat(
                    *[
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
                        for small_df in small_dfs
                    ],
                    context=ir_context,
                )

                # Send output chunk
                await ch_out.send(
                    context,
                    Message(
                        seq_num,
                        TableChunk.from_pylibcudf_table(
                            df.table, df.stream, exclusive_view=True
                        ),
                    ),
                )
                del df, large_df, large_chunk

        del small_dfs, small_chunks
        await ch_out.drain(context)


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
                collective_id=rec.state["collective_id_map"][ir][0],
                target_partition_size=target_partition_size,
            )
        ]
        return nodes, channels
