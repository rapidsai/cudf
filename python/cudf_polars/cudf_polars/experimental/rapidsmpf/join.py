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
    NormalizedPartitioning,
    allgather_reduce,
    chunk_to_frame,
    empty_table_chunk,
    names_to_indices,
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

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer

# Keep a conservative distance from the 2^31-1 (~2.15 billion) row limit
MAX_BROADCAST_ROWS = 1_500_000_000


@dataclass(frozen=True)
class JoinChunkSample:
    """Sampled chunks and aggregate size/row stats for one side of a join."""

    chunks: list[TableChunk]
    """The sampled chunks."""
    total_size: int
    """The total estimated size of the child table."""
    total_rows: int
    """The total estimated number of rows in the child table."""


@dataclass(frozen=True)
class JoinStrategy:
    """Summary of sampling and strategy selection for a dynamic join."""

    broadcast_side: Literal["left", "right"] | None
    """The side to broadcast. If None, the strategy is not a broadcast join."""
    min_shuffle_modulus: int
    """The minimum shuffle modulus."""
    left_sample: JoinChunkSample
    """Left-side sample information."""
    right_sample: JoinChunkSample
    """Right-side sample information."""


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
        context, ch_out, ch_left, ch_right, trace_ir=ir
    ) as tracer:
        left_metadata, right_metadata = await asyncio.gather(
            recv_metadata(ch_left, context),
            recv_metadata(ch_right, context),
        )
        await _broadcast_join(
            context,
            ir,
            ir_context,
            ch_out,
            ch_left,
            ch_right,
            left_metadata,
            right_metadata,
            [],  # No sampled chunks
            [],  # No sampled chunks
            broadcast_side,
            [collective_id],
            target_partition_size,
            tracer=tracer,
        )


async def _collect_small_side_for_broadcast(
    context: Context,
    small_ch: Channel[TableChunk],
    initial_chunks: list[TableChunk],
    small_child: IR,
    *,
    need_allgather: bool,
    collective_id: int,
    ir_context: Any,
    concat_size_limit: int | None,
) -> tuple[list[DataFrame], int]:
    """
    Drain small-side channel into chunks, then build DataFrame(s) for broadcast.

    Returns (list of DataFrames to join against, total byte size of small side).
    """
    small_chunks: list[TableChunk] = initial_chunks
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

    cudf_row_limit = 2**31 - 1
    if (can_concatenate := small_row_count < cudf_row_limit) and concat_size_limit:
        can_concatenate = small_size <= concat_size_limit
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
            small_dfs = [chunk_to_frame(c, small_child) for c in small_chunks]
        small_chunks.clear()

    return small_dfs, small_size


async def _broadcast_join_large_chunk(
    context: Context,
    ir: Join,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    small_dfs: list[DataFrame],
    small_child: IR,
    large_chunk: TableChunk,
    large_child: IR,
    seq_num: int,
    small_size: int,
    broadcast_side: Literal["left", "right"],
    *,
    tracer: ActorTracer | None,
) -> None:
    """Join one large-side chunk with the small DataFrame(s) and send the result."""
    large_df = chunk_to_frame(large_chunk, large_child)
    large_chunk_size = large_chunk.data_alloc_size(MemoryType.DEVICE)

    dfs_to_join = small_dfs
    if not dfs_to_join:
        stream = ir_context.get_cuda_stream()
        empty_small = empty_table_chunk(small_child, context, stream)
        dfs_to_join = [chunk_to_frame(empty_small, small_child)]

    join_results: list[DataFrame] = []
    input_bytes = large_chunk_size + small_size
    with opaque_memory_usage(
        await reserve_memory(context, size=input_bytes, net_memory_delta=0)
    ):
        for sdf in dfs_to_join:
            result = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                *([large_df, sdf] if broadcast_side == "right" else [sdf, large_df]),
                context=ir_context,
            )
            join_results.append(result)

        df = _concat(*join_results, context=ir_context)
        del join_results

    if tracer is not None:
        tracer.add_chunk(table=df.table)
    await ch_out.send(
        context,
        Message(
            seq_num,
            TableChunk.from_pylibcudf_table(df.table, df.stream, exclusive_view=True),
        ),
    )
    del df, large_df


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
    collective_ids: list[int],
    target_partition_size: int,
    *,
    tracer: ActorTracer | None,
) -> None:
    """
    Execute a broadcast join after initial sampling.

    The small side is gathered (if not already duplicated) and concatenated
    into a single DataFrame, then joined with each chunk from the large side.
    Pops one collective ID from collective_ids for allgather when needed.
    """
    collective_id = collective_ids.pop(0) if collective_ids else 0
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

    small_dfs, small_size = await _collect_small_side_for_broadcast(
        context,
        small_ch,
        small_initial_chunks,
        small_child,
        need_allgather=need_allgather,
        collective_id=collective_id,
        ir_context=ir_context,
        concat_size_limit=(target_partition_size if ir.options[0] == "Inner" else None),
    )

    # Stream through large side
    large_chunk_processed = False

    for seq_num, chunk in enumerate(large_initial_chunks):
        large_chunk_processed = True
        await _broadcast_join_large_chunk(
            context,
            ir,
            ir_context,
            ch_out,
            small_dfs,
            small_child,
            chunk,
            large_child,
            seq_num,
            small_size,
            broadcast_side,
            tracer=tracer,
        )

    while (msg := await large_ch.recv(context)) is not None:
        large_chunk_processed = True
        large_chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        msg_seq = msg.sequence_number

        await _broadcast_join_large_chunk(
            context,
            ir,
            ir_context,
            ch_out,
            small_dfs,
            small_child,
            large_chunk,
            large_child,
            msg_seq,
            small_size,
            broadcast_side,
            tracer=tracer,
        )
        del large_chunk

    if not large_chunk_processed and small_dfs:
        stream = ir_context.get_cuda_stream()
        large_chunk = empty_table_chunk(large_child, context, stream)
        await _broadcast_join_large_chunk(
            context,
            ir,
            ir_context,
            ch_out,
            small_dfs,
            small_child,
            large_chunk,
            large_child,
            0,
            small_size,
            broadcast_side,
            tracer=tracer,
        )
        del large_chunk

    del small_dfs
    await ch_out.drain(context)


def _get_key_indices(
    ir: Join,
    n_partitioned_keys: int | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    left, right = ir.children
    left_key_indices = names_to_indices(ir.left_on, left.schema)
    right_key_indices = names_to_indices(ir.right_on, right.schema)

    n_keys = (
        n_partitioned_keys if n_partitioned_keys is not None else len(left_key_indices)
    )
    output_schema_keys = list(ir.schema.keys())
    if ir.options == "Right":
        join_keys_for_output = ir.right_on
    else:
        join_keys_for_output = ir.left_on
    output_key_indices = tuple(
        output_schema_keys.index(expr.name)
        for expr in join_keys_for_output
        if expr.name in output_schema_keys
    )
    return (
        left_key_indices[:n_keys],
        right_key_indices[:n_keys],
        output_key_indices[:n_keys],
    )


def _create_shuffle_managers(
    context: Context,
    modulus: int,
    left_key_indices: tuple[int, ...],
    right_key_indices: tuple[int, ...],
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    collective_ids: list[int],
    tracer: ActorTracer | None,
) -> tuple[ShuffleManager | None, ShuffleManager | None]:
    left_partitioning_desired = NormalizedPartitioning(
        inter_rank_modulus=modulus,
        inter_rank_indices=left_key_indices,
        local_modulus=None,
        local_indices=(),
    )
    right_partitioning_desired = NormalizedPartitioning(
        inter_rank_modulus=modulus,
        inter_rank_indices=right_key_indices,
        local_modulus=None,
        local_indices=(),
    )

    shuffle_left = (
        not left_partitioning or left_partitioning != left_partitioning_desired
    )
    shuffle_right = (
        not right_partitioning or right_partitioning != right_partitioning_desired
    )

    left_shuffle = (
        ShuffleManager(
            context,
            modulus,
            left_partitioning_desired.inter_rank_indices,
            collective_ids.pop(0),
        )
        if shuffle_left
        else None
    )

    right_shuffle = (
        ShuffleManager(
            context,
            modulus,
            right_partitioning_desired.inter_rank_indices,
            collective_ids.pop(0),
        )
        if shuffle_right
        else None
    )

    if tracer is not None:
        if shuffle_left and shuffle_right:
            tracer.decision = "shuffle"
        elif shuffle_left:
            tracer.decision = "shuffle_left"
        elif shuffle_right:
            tracer.decision = "shuffle_right"
        else:
            tracer.decision = "chunkwise"

    return left_shuffle, right_shuffle


async def drain_into_shuffle(
    context: Context,
    ch: Channel[TableChunk],
    shuffle: ShuffleManager | None,
    sample_chunks: list[TableChunk],
) -> None:
    """Drain sample chunks and channel into a shuffle manager, then mark finished."""
    if shuffle is not None:
        while sample_chunks:
            shuffle.insert_chunk(
                sample_chunks.pop(0).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
            )
        while (msg := await ch.recv(context)) is not None:
            shuffle.insert_chunk(
                TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
            )
        await shuffle.insert_finished()


async def get_unshuffled_chunk(
    context: Context,
    ch: Channel[TableChunk],
    sample_chunks: list[TableChunk],
) -> TableChunk | None:
    """Return next chunk from sample list or channel."""
    if sample_chunks:
        return sample_chunks.pop(0)
    if (msg := await ch.recv(context)) is not None:
        return TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
    return None


async def get_dataframe_to_join(
    context: Context,
    ch: Channel[TableChunk],
    sample_chunks: list[TableChunk],
    *,
    shuffle: ShuffleManager | None,
    partition_id: int,
    stream: Stream,
    child: IR,
) -> DataFrame:
    """Get the next DataFrame (from shuffle or channel) for one side of the join."""
    if shuffle is not None:
        table = await shuffle.extract_chunk(partition_id, stream)
        if table.num_rows() == 0 and len(table.columns()) == 0:
            table = empty_table_chunk(child, context, stream).table_view()
    else:
        chunk = await get_unshuffled_chunk(context, ch, sample_chunks)
        if chunk is None:
            chunk = empty_table_chunk(child, context, stream)
        else:
            stream = chunk.stream
        table = chunk.table_view()

    return DataFrame.from_table(
        table, list(child.schema.keys()), list(child.schema.values()), stream
    )


async def _join_chunks(
    context: Context,
    ir: Join,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_sample_chunks: list[TableChunk],
    right_sample_chunks: list[TableChunk],
    left_shuffle: ShuffleManager | None,
    right_shuffle: ShuffleManager | None,
    *,
    partition_id: int,
    tracer: ActorTracer | None,
) -> None:
    left, right = ir.children
    stream = ir_context.get_cuda_stream()
    left_df, right_df = await asyncio.gather(
        get_dataframe_to_join(
            context,
            ch_left,
            left_sample_chunks,
            shuffle=left_shuffle,
            partition_id=partition_id,
            stream=stream,
            child=left,
        ),
        get_dataframe_to_join(
            context,
            ch_right,
            right_sample_chunks,
            shuffle=right_shuffle,
            partition_id=partition_id,
            stream=stream,
            child=right,
        ),
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
        del left_df, right_df
    if tracer is not None:
        tracer.add_chunk(table=df.table)
    await ch_out.send(
        context,
        Message(
            partition_id,
            TableChunk.from_pylibcudf_table(df.table, df.stream, exclusive_view=True),
        ),
    )
    del df


async def _shuffle_join(
    context: Context,
    ir: Join,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    min_shuffle_modulus: int,
    collective_ids: list[int],
    *,
    tracer: ActorTracer | None,
    n_partitioned_keys: int | None = None,
    left_sample_chunks: list[TableChunk] | None = None,
    right_sample_chunks: list[TableChunk] | None = None,
) -> None:
    """Execute a shuffle (hash) join."""
    modulus = _choose_shuffle_modulus(
        context,
        left_partitioning,
        right_partitioning,
        min_shuffle_modulus,
    )  # Global modulus

    left_key_indices, right_key_indices, output_key_indices = _get_key_indices(
        ir, n_partitioned_keys
    )

    local_count = max(1, modulus // context.comm().nranks)
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

    left_shuffle, right_shuffle = _create_shuffle_managers(
        context,
        modulus,
        left_key_indices,
        right_key_indices,
        left_partitioning,
        right_partitioning,
        collective_ids,
        tracer,
    )
    await asyncio.gather(
        drain_into_shuffle(context, ch_left, left_shuffle, left_sample_chunks or []),
        drain_into_shuffle(context, ch_right, right_shuffle, right_sample_chunks or []),
    )

    partition_ids: list[int]
    if left_shuffle is not None:
        partition_ids = left_shuffle.local_partitions()
    elif right_shuffle is not None:
        partition_ids = right_shuffle.local_partitions()
    else:
        partition_ids = list(range(local_count))
    for partition_id in partition_ids:
        await _join_chunks(
            context,
            ir,
            ir_context,
            ch_out,
            ch_left,
            ch_right,
            left_sample_chunks or [],
            right_sample_chunks or [],
            left_shuffle,
            right_shuffle,
            partition_id=partition_id,
            tracer=tracer,
        )

    del left_shuffle, right_shuffle
    await ch_out.drain(context)


async def _choose_strategy(
    context: Context,
    ir: Join,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    left_sample: JoinChunkSample,
    right_sample: JoinChunkSample,
    collective_ids: list[int],
    broadcast_threshold: int,
    target_partition_size: int,
) -> tuple[Literal["left", "right"] | None, int]:
    """Choose potential broadcast side and minimum shuffle modulus."""
    left_local_count = left_metadata.local_count
    right_local_count = right_metadata.local_count

    if left_sample.chunks:
        left_avg_size = left_sample.total_size / len(left_sample.chunks)
        left_avg_rows = left_sample.total_rows / len(left_sample.chunks)
        left_estimate = int(left_avg_size * left_local_count)
        left_row_estimate = int(left_avg_rows * left_local_count)
    else:
        left_estimate = 0
        left_row_estimate = 0

    if right_sample.chunks:
        right_avg_size = right_sample.total_size / len(right_sample.chunks)
        right_avg_rows = right_sample.total_rows / len(right_sample.chunks)
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

    # Determine which sides may be broadcasted
    left_size_ok = left_total < broadcast_threshold and (
        left_total_rows < MAX_BROADCAST_ROWS or left_metadata.duplicated
    )
    right_size_ok = right_total < broadcast_threshold and (
        right_total_rows < MAX_BROADCAST_ROWS or right_metadata.duplicated
    )
    can_broadcast_left = left_size_ok and ir.options[0] in ("Inner", "Right")
    can_broadcast_right = right_size_ok and ir.options[0] in (
        "Inner",
        "Left",
        "Semi",
        "Anti",
    )

    # Determine strategy
    broadcast_side: Literal["left", "right"] | None = None
    if can_broadcast_left and can_broadcast_right:
        # Choose side that is already duplicated.
        # If both or neither are duplicated, choose the side with fewer rows.
        if left_metadata.duplicated == right_metadata.duplicated:
            broadcast_side = "right" if right_total_rows <= left_total_rows else "left"
        elif left_metadata.duplicated:
            broadcast_side = "left"
        else:
            broadcast_side = "right"
    elif can_broadcast_left:
        broadcast_side = "left"
    elif can_broadcast_right:
        broadcast_side = "right"

    estimated_output_size = max(left_total, right_total)
    ideal_output_count = max(1, estimated_output_size // target_partition_size)
    # Limit the output count to 10x the larger input side
    max_output_chunks = 10 * max(left_total_chunks, right_total_chunks)
    min_shuffle_modulus = min(ideal_output_count, max_output_chunks)

    return broadcast_side, min_shuffle_modulus


def _choose_shuffle_modulus(
    context: Context,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    min_shuffle_modulus: int,
) -> int:
    """Choose an appropriate modulus for a shuffle join."""
    left_modulus = left_partitioning.inter_rank_modulus if left_partitioning else None
    right_modulus = (
        right_partitioning.inter_rank_modulus if right_partitioning else None
    )
    default_modulus = max(context.comm().nranks, min_shuffle_modulus)
    small, large = sorted(
        [left_modulus or default_modulus, right_modulus or default_modulus]
    )
    if large % small == 0:
        return small
    else:
        return large

    # # Determine which modulus to use, preferring existing partitioning
    # # if it provides at least the minimum needed partitions
    # if left_modulus and right_modulus:
    #     # Both sides partitioned - use the larger modulus if compatible
    #     # (one must be a multiple of the other for compatibility)
    #     if left_modulus >= right_modulus:
    #         if left_modulus % right_modulus == 0:
    #             modulus = left_modulus
    #         else:
    #             # Incompatible - use whichever is larger
    #             modulus = max(left_modulus, right_modulus)
    #     else:
    #         if right_modulus % left_modulus == 0:
    #             modulus = right_modulus
    #         else:
    #             modulus = max(left_modulus, right_modulus)
    # elif left_modulus is not None:
    #     # Only left is partitioned - use its modulus if sufficient
    #     modulus = left_modulus
    # elif right_modulus is not None:
    #     # Only right is partitioned - use its modulus if sufficient
    #     modulus = right_modulus
    # else:
    #     # Neither side partitioned - can choose freely
    #     # Use at least nranks for distributed efficiency
    #     modulus = context.comm().nranks
    # return max(modulus, min_shuffle_modulus)


async def _sample_chunks(
    context: Context,
    ch: Channel[TableChunk],
    sample_chunk_count: int,
) -> JoinChunkSample:
    """Sample up to sample_chunk_count chunks from a channel; return chunks and stats."""
    chunks: list[TableChunk] = []
    total_size = 0
    total_rows = 0
    for _ in range(sample_chunk_count):
        msg = await ch.recv(context)
        if msg is None:
            break
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        chunks.append(chunk)
        total_size += chunk.data_alloc_size(MemoryType.DEVICE)
        total_rows += chunk.table_view().num_rows()
    return JoinChunkSample(chunks=chunks, total_size=total_size, total_rows=total_rows)


async def _sample_and_choose_strategy(
    context: Context,
    ir: Join,
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    sample_chunk_count: int,
    broadcast_threshold: int,
    target_partition_size: int,
    collective_ids: list[int],
) -> JoinStrategy:
    """Sample both sides, allgather estimates, and choose broadcast vs shuffle."""
    left_sample, right_sample = await asyncio.gather(
        _sample_chunks(context, ch_left, sample_chunk_count),
        _sample_chunks(context, ch_right, sample_chunk_count),
    )
    broadcast_side, min_shuffle_modulus = await _choose_strategy(
        context,
        ir,
        left_metadata,
        right_metadata,
        left_sample,
        right_sample,
        collective_ids,
        broadcast_threshold,
        target_partition_size,
    )
    return JoinStrategy(
        broadcast_side=broadcast_side,
        min_shuffle_modulus=min_shuffle_modulus,
        left_sample=left_sample,
        right_sample=right_sample,
    )


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

    Receives metadata from the left and right channels, then either
    executes a shuffle join or a broadcast join. Strategy is chosen
    at runtime from sampled chunks when partitioning is not aligned.

    Parameters
    ----------
    context
        RapidsMPF context (communicator, etc.).
    ir
        The Join IR node.
    ir_context
        Execution context for the plan.
    ch_out
        Output channel for the join result.
    ch_left
        Input channel for the left side.
    ch_right
        Input channel for the right side.
    sample_chunk_count
        Number of chunks to sample per side for strategy selection.
    broadcast_threshold
        Max rows on one side to allow broadcast join (small side sent to all ranks).
    target_partition_size
        Target partition size used when choosing shuffle modulus.
    collective_ids
        List of collective IDs for shuffle/broadcast; consumed as needed.
    """
    async with shutdown_on_error(
        context, ch_out, ch_left, ch_right, trace_ir=ir
    ) as tracer:
        left_metadata, right_metadata = await asyncio.gather(
            recv_metadata(ch_left, context),
            recv_metadata(ch_right, context),
        )

        nranks = context.comm().nranks
        left_partitioning = NormalizedPartitioning.from_indices(
            left_metadata.partitioning,
            nranks,
            indices=names_to_indices(ir.left_on, ir.children[0].schema),
        )
        right_partitioning = NormalizedPartitioning.from_indices(
            right_metadata.partitioning,
            nranks,
            indices=names_to_indices(ir.right_on, ir.children[1].schema),
        )

        # Skip sampling when both sides have aligned partitioning
        if left_partitioning and left_partitioning == right_partitioning:
            await _shuffle_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                left_partitioning,
                right_partitioning,
                left_partitioning.inter_rank_modulus,
                collective_ids,
                n_partitioned_keys=len(left_partitioning.inter_rank_indices),
                tracer=tracer,
            )
            return

        strategy = await _sample_and_choose_strategy(
            context,
            ir,
            ch_left,
            ch_right,
            left_metadata,
            right_metadata,
            sample_chunk_count,
            broadcast_threshold,
            target_partition_size,
            collective_ids,
        )

        if strategy.broadcast_side is not None:
            await _broadcast_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                left_metadata,
                right_metadata,
                strategy.left_sample.chunks,
                strategy.right_sample.chunks,
                strategy.broadcast_side,
                collective_ids,
                target_partition_size,
                tracer=tracer,
            )
        else:
            await _shuffle_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                left_partitioning,
                right_partitioning,
                strategy.min_shuffle_modulus,
                collective_ids,
                left_sample_chunks=strategy.left_sample.chunks,
                right_sample_chunks=strategy.right_sample.chunks,
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
    executor = rec.state["config_options"].executor

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
