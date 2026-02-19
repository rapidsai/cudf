# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.memory_reserve_or_wait import (
    missing_net_memory_delta,
    reserve_memory,
)
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Join
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import default_node_multi
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    allgather_reduce,
    chunk_to_frame,
    empty_table_chunk,
    get_partitioning_moduli,
    process_children,
    recv_metadata,
    remap_partitioning,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.experimental.utils import _concat
from cudf_polars.utils.config import StreamingExecutor

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer

# Keep a conservative distance from the 2^31-1 (~2.15 billion) row limit
MAX_BROADCAST_ROWS = 1_500_000_000


def _is_partitioned_on_keys(
    metadata: ChannelMetadata,
    key_indices: tuple[int, ...],
    target_modulus: int,
) -> bool:
    """
    Check if data is already partitioned on the join keys with compatible modulus.

    Returns True if the data is partitioned on the specified key columns with
    a modulus that is a multiple of the target modulus (meaning it can be used
    directly or with local repartitioning).
    """
    if metadata.partitioning is None:
        return False
    inter_rank = metadata.partitioning.inter_rank
    if inter_rank is None or inter_rank == "inherit":
        return False
    # Check if partitioned on same keys
    if set(inter_rank.column_indices) != set(key_indices):
        return False
    # Check if modulus is compatible (existing modulus is a multiple of target)
    # This means the existing partitioning is at least as fine-grained
    return inter_rank.modulus % target_modulus == 0


# def _get_key_partitioning_modulus(
#     metadata: ChannelMetadata,
#     key_indices: tuple[int, ...],
#     nranks: int,
# ) -> int | None:
#     """
#     Get the modulus if data is partitioned on the specified keys.

#     Returns the modulus if partitioned on exactly the given keys, else None.
#     """
#     if metadata.partitioning is None:
#         return None
#     inter_rank = metadata.partitioning.inter_rank
#     local = metadata.partitioning.local
#     partitioning = local if (nranks == 0 and inter_rank is None) else inter_rank
#     if (
#         not isinstance(partitioning, HashScheme)
#         or local is None
#         or set(partitioning.column_indices) != set(key_indices)
#     ):
#         return None
#     return partitioning.modulus


@define_actor()
async def broadcast_join_actor(
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
    Broadcast-join actor for rapidsmpf.

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
    async with shutdown_on_error(
        context, ch_left, ch_right, ch_out, trace_ir=ir
    ) as tracer:
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
            # Remap partitioning from child schema to output schema
            partitioning = remap_partitioning(
                left_metadata.partitioning, large_child.schema, ir.schema
            )
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
                # Remap partitioning from child schema to output schema
                partitioning = remap_partitioning(
                    right_metadata.partitioning, large_child.schema, ir.schema
                )
            # Check if the right-side is already broadcasted
            small_duplicated = left_metadata.duplicated

        if tracer is not None:
            tracer.decision = f"broadcast_{broadcast_side}"

        # Determine which metadata belongs to the large side
        large_metadata = left_metadata if broadcast_side == "right" else right_metadata

        # Allgather is a collective - all ranks must participate even with no local data
        need_allgather = context.comm().nranks > 1 and not small_duplicated

        # The result is duplicated if:
        # - The small side is/will be duplicated (already duplicated OR will be AllGathered)
        # - AND the large side is already duplicated
        output_duplicated = (
            small_duplicated or need_allgather
        ) and large_metadata.duplicated

        # Send metadata.
        output_metadata = ChannelMetadata(
            local_count=local_count,
            partitioning=partitioning,
            duplicated=output_duplicated,
        )
        await send_metadata(ch_out, context, output_metadata)
        if tracer is not None:
            tracer.set_duplicated(duplicated=output_metadata.duplicated)

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
                large_chunk = await TableChunk.from_message(msg).make_available_or_wait(
                    context,
                    net_memory_delta=missing_net_memory_delta,
                )
                seq_num = msg.sequence_number

            large_df = DataFrame.from_table(
                large_chunk.table_view(),
                list(large_child.schema.keys()),
                list(large_child.schema.values()),
                large_chunk.stream,
            )
            large_chunk_size = large_chunk.data_alloc_size(MemoryType.DEVICE)
            del large_chunk  # `large_df` keeps `large_chunk` alive.

            # Lazily create empty small table if small_dfs is empty
            if not small_dfs:
                stream = ir_context.get_cuda_stream()
                empty_small_chunk = empty_table_chunk(small_child, context, stream)
                small_dfs = [chunk_to_frame(empty_small_chunk, small_child)]

            input_bytes = large_chunk_size + small_size
            with opaque_memory_usage(
                await reserve_memory(context, size=input_bytes, net_memory_delta=0)
            ):
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
                del large_df

            # Send output chunk
            output_chunk = TableChunk.from_pylibcudf_table(
                df.table, df.stream, exclusive_view=True
            )
            if tracer is not None:
                tracer.add_chunk(table=output_chunk.table_view())
            await ch_out.send(context, Message(seq_num, output_chunk))
            del df, output_chunk

        del small_dfs, small_chunks
        await ch_out.drain(context)


async def _broadcast_join(
    context: Context,
    ir: Join,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    left_sample_chunks: list[TableChunk],
    right_sample_chunks: list[TableChunk],
    broadcast_side: Literal["left", "right"],
    collective_id: int,
    target_partition_size: int,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Execute a broadcast join after initial sampling.

    The small side is gathered (if not already duplicated) and concatenated
    into a single DataFrame, then joined with each chunk from the large side.
    """
    left, right = ir.children
    if tracer is not None:
        tracer.decision = f"broadcast_{broadcast_side}"

    if broadcast_side == "right":
        small_ch, large_ch = ch_right, ch_left
        small_child, large_child = right, left
        small_metadata, large_metadata = right_metadata, left_metadata
        small_initial_chunks = right_sample_chunks
        large_initial_chunks = left_sample_chunks
        local_count = left_metadata.local_count
        partitioning: Partitioning | None = remap_partitioning(
            left_metadata.partitioning, large_child.schema, ir.schema
        )
    else:
        small_ch, large_ch = ch_left, ch_right
        small_child, large_child = left, right
        small_metadata, large_metadata = left_metadata, right_metadata
        small_initial_chunks = left_sample_chunks
        large_initial_chunks = right_sample_chunks
        local_count = right_metadata.local_count
        partitioning = (
            remap_partitioning(
                right_metadata.partitioning, large_child.schema, ir.schema
            )
            if ir.options[0] == "Right"
            else None
        )

    small_duplicated = small_metadata.duplicated
    need_allgather = context.comm().nranks > 1 and not small_duplicated
    output_duplicated = (
        small_duplicated or need_allgather
    ) and large_metadata.duplicated

    metadata_out = ChannelMetadata(
        local_count=local_count,
        partitioning=partitioning,
        duplicated=output_duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    # Collect remaining small-side chunks
    small_chunks: list[TableChunk] = list(small_initial_chunks)
    small_size = sum(c.data_alloc_size(MemoryType.DEVICE) for c in small_chunks)
    small_row_count = sum(c.table_view().num_rows() for c in small_chunks)
    while (msg := await small_ch.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        small_chunks.append(chunk)
        small_size += chunk.data_alloc_size(MemoryType.DEVICE)
        small_row_count += chunk.table_view().num_rows()
        del msg

    # Check if we can safely concatenate the small side
    # cuDF column limit is 2^31-1 = 2,147,483,647 rows
    cudf_row_limit = 2**31 - 1
    can_concatenate = small_row_count < cudf_row_limit

    # Build small-side DataFrame list
    # If row count is safe, concatenate into single-element list for efficiency
    # Otherwise, keep as list of individual DataFrames
    small_dfs: list[DataFrame] = []

    if need_allgather:
        allgather = AllGatherManager(context, collective_id)
        for s_id in range(len(small_chunks)):
            allgather.insert(s_id, small_chunks.pop(0))
        allgather.insert_finished()
        stream = ir_context.get_cuda_stream()
        small_dfs = [
            DataFrame.from_table(
                await allgather.extract_concatenated(stream),
                list(small_child.schema.keys()),
                list(small_child.schema.values()),
                stream,
            )
        ]
    elif small_chunks:
        if can_concatenate:
            # Safe to concatenate - produces single-element list
            # (_concat is a no-op for single element)
            # Reserve memory for concatenation (input + output ~= 2x input size)
            small_chunks, extra = await make_table_chunks_available_or_wait(
                context,
                small_chunks,
                reserve_extra=small_size,
                net_memory_delta=0,
            )
            with opaque_memory_usage(extra):
                small_dfs = [
                    _concat(
                        *[chunk_to_frame(chunk, small_child) for chunk in small_chunks],
                        context=ir_context,
                    )
                ]
        else:
            # Too many rows to concatenate - keep as list of DataFrames
            small_dfs = [chunk_to_frame(c, small_child) for c in small_chunks]
        small_chunks.clear()

    # Stream through large side
    large_chunk_processed = False

    async def join_large_chunk(
        large_df: DataFrame, seq_num: int, large_chunk_size: int
    ) -> None:
        """Join a large chunk with the small DataFrame(s) and send result."""
        nonlocal small_dfs

        # Get small DataFrames to join with (create empty if none)
        dfs_to_join = small_dfs
        if not dfs_to_join:
            stream = ir_context.get_cuda_stream()
            empty_small = empty_table_chunk(small_child, context, stream)
            dfs_to_join = [chunk_to_frame(empty_small, small_child)]

        # Join large chunk with each small DataFrame
        join_results: list[DataFrame] = []
        input_bytes = large_chunk_size + small_size
        with opaque_memory_usage(
            await reserve_memory(context, size=input_bytes, net_memory_delta=0)
        ):
            for sdf in dfs_to_join:
                result = await asyncio.to_thread(
                    ir.do_evaluate,
                    *ir._non_child_args,
                    *(
                        [large_df, sdf]
                        if broadcast_side == "right"
                        else [sdf, large_df]
                    ),
                    context=ir_context,
                )
                join_results.append(result)

            # Concatenate join results (_concat is no-op for single element)
            df = _concat(*join_results, context=ir_context)
            del join_results

        if tracer is not None:
            tracer.add_chunk(table=df.table)
        await ch_out.send(
            context,
            Message(
                seq_num,
                TableChunk.from_pylibcudf_table(
                    df.table, df.stream, exclusive_view=True
                ),
            ),
        )
        del df

    # Process initial large chunks first
    for seq_num, chunk in enumerate(large_initial_chunks):
        large_chunk_processed = True
        large_df = DataFrame.from_table(
            chunk.table_view(),
            list(large_child.schema.keys()),
            list(large_child.schema.values()),
            chunk.stream,
        )
        await join_large_chunk(
            large_df, seq_num, chunk.data_alloc_size(MemoryType.DEVICE)
        )
        del large_df

    # Process remaining large chunks from channel
    while (msg := await large_ch.recv(context)) is not None:
        large_chunk_processed = True
        large_chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        msg_seq = msg.sequence_number
        del msg

        large_df = DataFrame.from_table(
            large_chunk.table_view(),
            list(large_child.schema.keys()),
            list(large_child.schema.values()),
            large_chunk.stream,
        )
        await join_large_chunk(
            large_df, msg_seq, large_chunk.data_alloc_size(MemoryType.DEVICE)
        )
        del large_df, large_chunk

    # Handle edge case: no large-side data received
    if not large_chunk_processed and small_dfs:
        stream = ir_context.get_cuda_stream()
        large_chunk = empty_table_chunk(large_child, context, stream)
        large_df = chunk_to_frame(large_chunk, large_child)
        await join_large_chunk(large_df, 0, 0)
        del large_df

    del small_dfs, small_chunks
    await ch_out.drain(context)


async def _shuffle_join(
    context: Context,
    ir: Join,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_state: PartitioningState,
    right_state: PartitioningState,
    min_shuffle_modulus: int,
    collective_ids: list[int],
    *,
    n_partitioned_keys: int | None = None,
    left_sample_chunks: list[TableChunk] | None = None,
    right_sample_chunks: list[TableChunk] | None = None,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Execute a shuffle (hash) join.

    Desired states are built inside from modulus (and n_partitioned_keys when
    provided). Shuffles a side iff its current state does not match. Output
    metadata reflects the desired partitioning (may be a subset of join keys).
    """
    nranks = context.comm().nranks
    modulus = _choose_shuffle_modulus(
        left_state,
        right_state,
        min_shuffle_modulus,
        nranks,
    )

    left, right = ir.children
    left_schema_keys = list(left.schema.keys())
    right_schema_keys = list(right.schema.keys())
    left_key_indices = tuple(left_schema_keys.index(expr.name) for expr in ir.left_on)
    right_key_indices = tuple(
        right_schema_keys.index(expr.name) for expr in ir.right_on
    )

    n_keys = (
        n_partitioned_keys if n_partitioned_keys is not None else len(left_key_indices)
    )
    left_desired_state = PartitioningState(
        inter_rank_modulus=modulus,
        inter_rank_indices=left_key_indices[:n_keys],
        local_modulus=None,
        local_indices=None,
    )
    right_desired_state = PartitioningState(
        inter_rank_modulus=modulus,
        inter_rank_indices=right_key_indices[:n_keys],
        local_modulus=None,
        local_indices=None,
    )

    shuffle_left = not left_state or left_state != left_desired_state
    shuffle_right = not right_state or right_state != right_desired_state

    local_count = max(1, modulus // nranks)

    if tracer is not None:
        if shuffle_left and shuffle_right:
            tracer.decision = "shuffle"
        elif shuffle_left:
            tracer.decision = "shuffle_left"
        elif shuffle_right:
            tracer.decision = "shuffle_right"
        else:
            tracer.decision = "chunkwise"

    output_schema_keys = list(ir.schema.keys())
    if ir.options[0] in ("Inner", "Left", "Full", "Semi", "Anti"):
        join_keys_for_output = ir.left_on
    else:
        join_keys_for_output = ir.right_on
    output_key_indices = tuple(
        output_schema_keys.index(expr.name)
        for expr in join_keys_for_output[:n_keys]
        if expr.name in output_schema_keys
    )

    metadata_out = ChannelMetadata(
        local_count=local_count,
        partitioning=Partitioning(
            HashScheme(column_indices=output_key_indices, modulus=modulus),
            local="inherit",
        )
        if output_key_indices
        else None,
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    left_shuffle: ShuffleManager | None = None
    if shuffle_left:
        left_shuffle = ShuffleManager(
            context,
            modulus,
            left_desired_state.inter_rank_indices,
            collective_ids.pop(0),
        )
        while left_sample_chunks:
            left_shuffle.insert_chunk(
                left_sample_chunks.pop(0).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
            )

    right_shuffle: ShuffleManager | None = None
    if shuffle_right:
        right_shuffle = ShuffleManager(
            context,
            modulus,
            right_desired_state.inter_rank_indices,
            collective_ids.pop(0),
        )
        while right_sample_chunks:
            right_shuffle.insert_chunk(
                right_sample_chunks.pop(0).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
            )

    async def drain_shuffled_left() -> None:
        nonlocal left_shuffle
        if shuffle_left:
            assert left_shuffle is not None
            while (msg := await ch_left.recv(context)) is not None:
                left_shuffle.insert_chunk(
                    TableChunk.from_message(msg).make_available_and_spill(
                        context.br(), allow_overbooking=True
                    )
                )
                del msg
            await left_shuffle.insert_finished()

    async def drain_shuffled_right() -> None:
        nonlocal right_shuffle
        if shuffle_right:
            assert right_shuffle is not None
            while (msg := await ch_right.recv(context)) is not None:
                right_shuffle.insert_chunk(
                    TableChunk.from_message(msg).make_available_and_spill(
                        context.br(), allow_overbooking=True
                    )
                )
                del msg
            await right_shuffle.insert_finished()

    await asyncio.gather(drain_shuffled_left(), drain_shuffled_right())

    left_channel_done = shuffle_left

    async def get_left_chunk(
        left_sample_chunks: list[TableChunk] | None,
    ) -> TableChunk | None:
        nonlocal left_channel_done
        if left_sample_chunks:
            return left_sample_chunks.pop(0)
        if not left_channel_done:
            msg = await ch_left.recv(context)
            if msg is None:
                left_channel_done = True
                return None
            chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            del msg
            return chunk
        return None

    right_channel_done = shuffle_right

    async def get_right_chunk(
        right_sample_chunks: list[TableChunk] | None,
    ) -> TableChunk | None:
        nonlocal right_channel_done
        if right_sample_chunks:
            return right_sample_chunks.pop(0)
        if not right_channel_done:
            msg = await ch_right.recv(context)
            if msg is None:
                right_channel_done = True
                return None
            chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            del msg
            return chunk
        return None

    partition_ids: list[int]
    if left_shuffle is not None:
        partition_ids = left_shuffle.local_partitions()
    elif right_shuffle is not None:
        partition_ids = right_shuffle.local_partitions()
    else:
        partition_ids = list(range(local_count))

    stream = ir_context.get_cuda_stream()
    for seq_num, partition_id in enumerate(partition_ids):
        if left_shuffle is not None:
            left_table = await left_shuffle.extract_chunk(partition_id, stream)
        else:
            chunk = await get_left_chunk(left_sample_chunks)
            if chunk is not None:
                left_table = chunk.table_view()
            else:
                left_table = empty_table_chunk(left, context, stream).table_view()

        if right_shuffle is not None:
            right_table = await right_shuffle.extract_chunk(partition_id, stream)
        else:
            chunk = await get_right_chunk(right_sample_chunks)
            if chunk is not None:
                right_table = chunk.table_view()
            else:
                right_table = empty_table_chunk(right, context, stream).table_view()

        left_df = DataFrame.from_table(
            left_table, list(left.schema.keys()), list(left.schema.values()), stream
        )
        right_df = DataFrame.from_table(
            right_table, list(right.schema.keys()), list(right.schema.values()), stream
        )

        input_bytes = sum(
            col.device_buffer_size()
            for col in (*left_df.table.columns(), *right_df.table.columns())
        )
        with opaque_memory_usage(
            await reserve_memory(context, size=input_bytes, net_memory_delta=0)
        ):
            df = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                left_df,
                right_df,
                context=ir_context,
            )
            del left_df, right_df, left_table, right_table
        if tracer is not None:
            tracer.add_chunk(table=df.table)
        await ch_out.send(
            context,
            Message(
                seq_num,
                TableChunk.from_pylibcudf_table(
                    df.table, df.stream, exclusive_view=True
                ),
            ),
        )
        del df

    del left_shuffle, right_shuffle
    await ch_out.drain(context)


@dataclass(frozen=True)
class PartitioningState:
    """Partitioning state derived from channel metadata and key indices."""

    inter_rank_modulus: int
    """Inter-rank hash modulus (0 if not partitioned between ranks)."""
    inter_rank_indices: tuple[int, ...]
    """Table column indices that are inter-rank partitioned."""
    local_modulus: int | None
    """Local hash modulus (0 if not partitioned within rank, None if inherited)."""
    local_indices: tuple[int, ...] | None
    """Table column indices that are locally partitioned, or None if inherited."""

    def __bool__(self) -> bool:
        """True if there is effective inter-rank partitioning."""
        return self.inter_rank_modulus > 0

    def __eq__(self, other: object) -> bool:
        """Equal when moduli and same number of partitioned keys (for left vs right)."""
        if not isinstance(other, PartitioningState):
            return NotImplemented
        return (
            self.inter_rank_modulus == other.inter_rank_modulus
            and self.local_modulus == other.local_modulus
            and len(self.inter_rank_indices) == len(other.inter_rank_indices)
            and len(self.local_indices or ()) == len(other.local_indices or ())
        )

    @property
    def is_simply_partitioned(self) -> bool:
        """
        Whether the data is simply partitioned.

        Notes
        -----
        Return True if the data is partitioned and the
        total number of chunks matches the inter-rank modulus.
        """
        return self.inter_rank_modulus != 0 and self.local_modulus is None

    @classmethod
    def from_metadata(
        cls,
        metadata: ChannelMetadata,
        key_indices: tuple[int, ...],
        nranks: int,
    ) -> PartitioningState:
        """Build state from channel metadata and key indices (allow_subset=True)."""
        inter_rank_modulus, local_modulus = get_partitioning_moduli(
            metadata, key_indices, nranks, allow_subset=True
        )

        local_indices: tuple[int, ...] | None = None
        inter_rank_indices: tuple[int, ...] = key_indices
        if inter_rank_modulus and metadata.partitioning is not None:
            inter_rank_hashed = isinstance(metadata.partitioning.inter_rank, HashScheme)
            local_hashed = isinstance(metadata.partitioning.local, HashScheme)
            if local_hashed and inter_rank_hashed:
                inter_rank_indices = metadata.partitioning.inter_rank.column_indices
                local_indices = metadata.partitioning.local.column_indices
            elif inter_rank_hashed:
                inter_rank_indices = metadata.partitioning.inter_rank.column_indices
            elif local_hashed:
                if nranks == 1 and local_modulus:
                    inter_rank_modulus = local_modulus
                    inter_rank_indices = metadata.partitioning.local.column_indices
                    local_modulus = None
                else:
                    local_indices = metadata.partitioning.local.column_indices

        return cls(
            inter_rank_modulus=inter_rank_modulus,
            inter_rank_indices=inter_rank_indices,
            local_modulus=local_modulus,
            local_indices=local_indices or (),
        )


def _extract_partitioning_state(
    metadata: ChannelMetadata,
    ir: IR,
    on: tuple[NamedExpr, ...],
    nranks: int,
) -> PartitioningState:
    schema_keys = list(ir.schema.keys())
    key_indices = tuple(schema_keys.index(expr.name) for expr in on)
    return PartitioningState.from_metadata(metadata, key_indices, nranks)


async def _choose_algorithm(
    context: Context,
    ir: Join,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    left_sample_chunks: list[TableChunk],
    right_sample_chunks: list[TableChunk],
    left_sample_size: int,
    right_sample_size: int,
    left_sample_rows: int,
    right_sample_rows: int,
    collective_ids: list[int],
    broadcast_threshold: int,
    target_partition_size: int,
) -> tuple[Literal["left", "right"] | None, int]:
    # Estimate total sizes and row counts
    left_local_count = left_metadata.local_count
    right_local_count = right_metadata.local_count

    if left_sample_chunks:
        left_avg_size = left_sample_size / len(left_sample_chunks)
        left_avg_rows = left_sample_rows / len(left_sample_chunks)
        left_estimate = int(left_avg_size * left_local_count)
        left_row_estimate = int(left_avg_rows * left_local_count)
    else:
        left_estimate = 0
        left_row_estimate = 0

    if right_sample_chunks:
        right_avg_size = right_sample_size / len(right_sample_chunks)
        right_avg_rows = right_sample_rows / len(right_sample_chunks)
        right_estimate = int(right_avg_size * right_local_count)
        right_row_estimate = int(right_avg_rows * right_local_count)
    else:
        right_estimate = 0
        right_row_estimate = 0

    # AllGather size, row, and chunk count estimates across ranks
    nranks = context.comm().nranks
    if nranks > 1:
        (
            left_total,
            right_total,
            left_total_rows,
            right_total_rows,
            left_total_chunks,
            right_total_chunks,
        ) = await allgather_reduce(
            context,
            collective_ids.pop(),
            left_estimate,
            right_estimate,
            left_row_estimate,
            right_row_estimate,
            left_local_count,
            right_local_count,
        )
    else:
        left_total, right_total = left_estimate, right_estimate
        left_total_rows, right_total_rows = left_row_estimate, right_row_estimate
        left_total_chunks, right_total_chunks = left_local_count, right_local_count

    # =====================================================================
    # Strategy Selection
    # =====================================================================
    # - Inner: can broadcast either side
    # - Left/Semi/Anti: must broadcast right (stream left to preserve all left rows)
    # - Right: must broadcast left (stream right to preserve all right rows)
    # - Full: cannot broadcast (must shuffle both to preserve both sides)

    join_type = ir.options[0]
    can_broadcast_left = join_type in ("Inner", "Right")
    can_broadcast_right = join_type in ("Inner", "Left", "Semi", "Anti")

    # Check if one side is already duplicated
    left_duplicated = left_metadata.duplicated
    right_duplicated = right_metadata.duplicated

    # Check row counts - can't broadcast if concatenated rows exceed cuDF limit
    left_rows_ok = left_total_rows < MAX_BROADCAST_ROWS
    right_rows_ok = right_total_rows < MAX_BROADCAST_ROWS

    # Determine strategy
    broadcast_side: Literal["left", "right"] | None = None

    if nranks == 1:
        # Single rank - no network cost, but still prefer smaller side for
        # hash table efficiency. Also check broadcast threshold.
        left_ok = (
            left_total < broadcast_threshold and left_rows_ok and can_broadcast_left
        )
        right_ok = right_total < broadcast_threshold and right_rows_ok

        if left_ok and right_ok:
            # Both sides OK - broadcast the side with fewer rows
            # Row count is a better indicator of hash table size than byte size
            broadcast_side = (
                "left"
                if (not can_broadcast_right or left_total_rows <= right_total_rows)
                else "right"
            )
        elif right_ok and can_broadcast_right:
            broadcast_side = "right"
        elif left_ok:
            broadcast_side = "left"
        # else: fall through to shuffle
    elif right_duplicated and right_rows_ok and can_broadcast_right:
        # Right already duplicated - broadcast right (no allgather needed)
        broadcast_side = "right"
    elif left_duplicated and can_broadcast_left and left_rows_ok:
        # Left already duplicated - broadcast left (only for Inner)
        broadcast_side = "left"
    else:
        # Decide based on size estimates - broadcast the smaller side if possible
        # Must also check row counts to avoid exceeding cuDF column size limit
        left_small_enough = (
            left_total < broadcast_threshold and can_broadcast_left and left_rows_ok
        )
        right_small_enough = right_total < broadcast_threshold and right_rows_ok

        if left_small_enough and right_small_enough:
            # Both are small enough - broadcast the smaller one (if both allowed)
            broadcast_side = (
                "left"
                if (not can_broadcast_right or left_total <= right_total)
                else "right"
            )
        elif right_small_enough and can_broadcast_right:
            # Only right is small enough
            broadcast_side = "right"
        elif left_small_enough:
            # Only left is small enough (Inner or Right join)
            broadcast_side = "left"
        # else: shuffle both sides

    estimated_output_size = max(left_total, right_total)
    ideal_output_count = max(1, estimated_output_size // target_partition_size)
    ideal_modulus = nranks * ideal_output_count
    # Limit the output count to 10x the larger input side
    max_output_chunks = 10 * max(left_total_chunks, right_total_chunks)
    min_shuffle_modulus = min(ideal_modulus, max_output_chunks)

    return broadcast_side, min_shuffle_modulus


def _choose_shuffle_modulus(
    left_state: PartitioningState,
    right_state: PartitioningState,
    min_shuffle_modulus: int,
    nranks: int,
) -> int:
    left_existing_modulus = left_state.inter_rank_modulus if left_state else None
    right_existing_modulus = right_state.inter_rank_modulus if right_state else None

    # Determine which modulus to use, preferring existing partitioning
    # if it provides at least the minimum needed partitions
    if left_existing_modulus and right_existing_modulus:
        # Both sides partitioned - use the larger modulus if compatible
        # (one must be a multiple of the other for compatibility)
        if left_existing_modulus >= right_existing_modulus:
            if left_existing_modulus % right_existing_modulus == 0:
                modulus = left_existing_modulus
            else:
                # Incompatible - use whichever is larger
                modulus = max(left_existing_modulus, right_existing_modulus)
        else:
            if right_existing_modulus % left_existing_modulus == 0:
                modulus = right_existing_modulus
            else:
                modulus = max(left_existing_modulus, right_existing_modulus)
    elif left_existing_modulus is not None:
        # Only left is partitioned - use its modulus if sufficient
        modulus = max(left_existing_modulus, min_shuffle_modulus)
    elif right_existing_modulus is not None:
        # Only right is partitioned - use its modulus if sufficient
        modulus = max(right_existing_modulus, min_shuffle_modulus)
    else:
        # Neither side partitioned - can choose freely
        # Use at least nranks for distributed efficiency
        modulus = max(nranks, min_shuffle_modulus)
    return modulus


async def _sample_chunks(
    context: Context,
    ch: Channel[TableChunk],
    sample_chunk_count: int,
) -> tuple[list[TableChunk], int, int]:
    sample_chunks: list[TableChunk] = []
    sample_size = 0
    sample_rows = 0
    for _ in range(sample_chunk_count):
        msg = await ch.recv(context)
        if msg is None:
            break
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        sample_chunks.append(chunk)
        sample_size += chunk.data_alloc_size(MemoryType.DEVICE)
        sample_rows += chunk.table_view().num_rows()
        del msg
    return sample_chunks, sample_size, sample_rows


@define_actor()
async def join_actor(
    context: Context,
    ir: Join,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    sample_chunk_count: int,
    broadcast_threshold: int,
    target_partition_size: int,
    collective_ids: list[int],
) -> None:
    """
    Dynamic Join actor that selects the best strategy at runtime.

    Strategy selection based on sampled data:
    - Broadcast right: If right side is small (< broadcast_threshold)
    - Broadcast left: If left side is small (< broadcast_threshold)
    - Shuffle: Both sides are large, shuffle by join keys
    """
    async with shutdown_on_error(
        context, ch_left, ch_right, ch_out, trace_ir=ir
    ) as tracer:
        left_metadata, right_metadata = await asyncio.gather(
            recv_metadata(ch_left, context),
            recv_metadata(ch_right, context),
        )

        nranks = context.comm().nranks
        left_state = _extract_partitioning_state(
            left_metadata, ir.children[0], ir.left_on, nranks
        )
        right_state = _extract_partitioning_state(
            right_metadata, ir.children[1], ir.right_on, nranks
        )

        # Skip sampling when both sides have aligned partitioning
        if left_state and left_state == right_state:
            await _shuffle_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                left_state,
                right_state,
                left_state.inter_rank_modulus,
                collective_ids,
                n_partitioned_keys=len(left_state.inter_rank_indices),
                tracer=tracer,
            )
            return

        # Sample chunks from both sides concurrently
        (
            (left_sample_chunks, left_sample_size, left_sample_rows),
            (right_sample_chunks, right_sample_size, right_sample_rows),
        ) = await asyncio.gather(
            _sample_chunks(context, ch_left, sample_chunk_count),
            _sample_chunks(context, ch_right, sample_chunk_count),
        )

        broadcast_side, min_shuffle_modulus = await _choose_algorithm(
            context,
            ir,
            left_metadata,
            right_metadata,
            left_sample_chunks,
            right_sample_chunks,
            left_sample_size,
            right_sample_size,
            left_sample_rows,
            right_sample_rows,
            collective_ids,
            broadcast_threshold,
            target_partition_size,
        )

        if broadcast_side:
            await _broadcast_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                left_metadata,
                right_metadata,
                left_sample_chunks,
                right_sample_chunks,
                broadcast_side,
                collective_ids.pop(0),
                target_partition_size,
                tracer,
            )
        else:
            await _shuffle_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                left_state,
                right_state,
                min_shuffle_modulus,
                collective_ids,
                left_sample_chunks=left_sample_chunks,
                right_sample_chunks=right_sample_chunks,
                tracer=tracer,
            )


@generate_ir_sub_network.register(Join)
def _(
    ir: Join, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Join operation.
    left, right = ir.children
    partition_info = rec.state["partition_info"]
    output_count = partition_info[ir].count
    config_options = rec.state["config_options"]
    executor = config_options.executor

    # Check for dynamic planning
    join_type = ir.options[0]
    use_dynamic = (
        isinstance(executor, StreamingExecutor)
        and executor.dynamic_planning is not None
        and join_type in ("Inner", "Left", "Right", "Full", "Semi", "Anti")
    )

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
    actors, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    if pwise_join:
        # Partition-wise join (use default_node_multi)
        partitioning_index = 1 if ir.options[0] == "Right" else 0
        actors[ir] = [
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
        return actors, channels

    elif use_dynamic:
        # Dynamic join - decide strategy at runtime
        assert executor.dynamic_planning is not None  # Checked in use_dynamic
        collective_ids = list(rec.state["collective_id_map"].get(ir, []))
        # Join uses up to 3 collective IDs: 1 allgather + up to 2 (left/right shuffle)
        if len(collective_ids) < 3:
            raise ValueError(
                "Dynamic join requires 3 reserved collective IDs "
                "(allgather + left shuffle + right shuffle); got "
                f"{len(collective_ids)} for this Join. "
                "Ensure ReserveOpIDs is run with dynamic_planning enabled."
            )
        broadcast_threshold = (
            executor.target_partition_size * executor.broadcast_join_limit
        )
        actors[ir] = [
            join_actor(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                executor.dynamic_planning.sample_chunk_count,
                broadcast_threshold,
                executor.target_partition_size,
                collective_ids,
            )
        ]
        return actors, channels

    else:
        # Broadcast join (use broadcast_join_actor)
        broadcast_side: Literal["left", "right"]
        if left_count >= right_count:
            # Broadcast right, stream left
            broadcast_side = "right"
        else:
            broadcast_side = "left"
        actors[ir] = [
            broadcast_join_actor(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                broadcast_side=broadcast_side,
                collective_id=rec.state["collective_id_map"][ir][0],
                target_partition_size=executor.target_partition_size,
            )
        ]
        return actors, channels
