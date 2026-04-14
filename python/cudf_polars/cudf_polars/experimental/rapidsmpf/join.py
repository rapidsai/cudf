# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.memory_reserve_or_wait import (
    reserve_memory,
)
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.bloom_filter import BloomFilter
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Join
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import _global_shuffle
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
    make_spill_function,
    gather_in_task_group,
    maybe_remap_partitioning,
    names_to_indices,
    process_children,
    recv_metadata,
    replay_buffered_channel,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer
    from cudf_polars.utils.config import StreamingExecutor


# cuDF column/concatenate row limit (int32)
CUDF_ROW_LIMIT = 2**31 - 1
MAX_BROADCAST_ROWS = CUDF_ROW_LIMIT // 2

# Bloom filter L2 cache size for sizing the filter (32 MB)
_BLOOM_L2_SIZE = 32 * 1024 * 1024


@dataclass(frozen=True)
class JoinSideStats:
    """Sampled chunks and aggregate size/row stats for one side of a join."""

    chunks: dict[int, TableChunk] = field(default_factory=dict)
    """The sampled chunks, keyed by sequence number."""
    total_size: int = 0
    """The total estimated size of the child table."""
    total_rows: int = 0
    """The total estimated number of rows in the child table."""
    total_chunks: int = 0
    """The total estimated number of chunks in the child table."""


@dataclass(frozen=True)
class JoinStrategy:
    """Summary of sampling and strategy selection for a dynamic join."""

    broadcast_side: Literal["left", "right"] | None = None
    """The side to broadcast. If None, the strategy is a shuffle join."""
    shuffle_modulus: int = 0
    """The shuffle modulus. Only used for shuffle joins."""
    output_indices: tuple[int, ...] = ()
    """The shuffle indices for the output. Only used for shuffle joins."""
    left_indices: tuple[int, ...] = ()
    """The shuffle indices for the left side. Only used for shuffle joins."""
    right_indices: tuple[int, ...] = ()
    """The shuffle indices for the right side. Only used for shuffle joins."""


@define_actor()
async def broadcast_join_actor(
    context: Context,
    comm: Communicator,
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
    comm
        The communicator.
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
        context,
        ch_out,
        ch_left,
        ch_right,
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        await _broadcast_join(
            context,
            comm,
            ir,
            ir_context,
            ch_out,
            ch_left,
            ch_right,
            JoinStrategy(broadcast_side=broadcast_side),
            [collective_id],
            target_partition_size,
            tracer=tracer,
        )


async def _collect_small_side_for_broadcast(
    context: Context,
    comm: Communicator,
    ch: Channel[TableChunk],
    ir: IR,
    *,
    need_allgather: bool,
    collective_id: int,
    ir_context: IRExecutionContext,
    concat_size_limit: int | None,
) -> tuple[list[DataFrame], int]:
    """
    Drain small-side channel into chunks, then build DataFrame(s) for broadcast.

    Returns (list of DataFrames to join against, total byte size of small side).
    """
    size = 0
    chunks: list[TableChunk] = []
    while (msg := await ch.recv(context)) is not None:
        chunks.append(TableChunk.from_message(msg, br=context.br()))
        size += chunks[-1].data_alloc_size()
    row_count = sum(c.shape[0] for c in chunks)

    if (can_concatenate := row_count < CUDF_ROW_LIMIT) and concat_size_limit:
        can_concatenate = size <= concat_size_limit

    dfs: list[DataFrame] = []
    if need_allgather:
        allgather = AllGatherManager(context, comm, collective_id)
        for s_id in range(len(chunks)):
            allgather.insert(s_id, chunks.pop(0))
        allgather.insert_finished()
        stream = ir_context.get_cuda_stream()
        dfs = [
            DataFrame.from_table(
                await allgather.extract_concatenated(stream),
                list(ir.schema.keys()),
                list(ir.schema.values()),
                stream,
            )
        ]
    elif chunks:
        if can_concatenate:
            chunks, extra = await make_table_chunks_available_or_wait(
                context,
                chunks,
                reserve_extra=size,
                net_memory_delta=0,
            )
            with opaque_memory_usage(extra):
                dfs = [
                    _concat(
                        *[chunk_to_frame(chunk, ir) for chunk in chunks],
                        context=ir_context,
                    )
                ]
        else:
            chunks, _ = await make_table_chunks_available_or_wait(
                context, chunks, reserve_extra=0, net_memory_delta=0
            )
            dfs = [chunk_to_frame(c, ir) for c in chunks]

    return dfs, size


async def _broadcast_join_large_chunk(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
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
    large_chunk_size = large_chunk.data_alloc_size()

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
            TableChunk.from_pylibcudf_table(
                df.table, df.stream, exclusive_view=True, br=context.br()
            ),
        ),
    )
    del df, large_df


async def _broadcast_join(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    strategy: JoinStrategy,
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
    left_metadata, right_metadata = await gather_in_task_group(
        recv_metadata(ch_left, context),
        recv_metadata(ch_right, context),
    )

    collective_id = collective_ids.pop(0) if collective_ids else 0
    broadcast_side = strategy.broadcast_side
    assert broadcast_side is not None
    left, right = ir.children
    if tracer is not None:
        tracer.decision = f"broadcast_{broadcast_side}"

    if broadcast_side == "right":
        small_ch, large_ch = ch_right, ch_left
        small_child, large_child = right, left
        small_metadata, large_metadata = right_metadata, left_metadata
        local_count = left_metadata.local_count
        partitioning = maybe_remap_partitioning(
            ir,
            left_metadata.partitioning,
            child_ir=ir.children[0],
        )
    else:
        small_ch, large_ch = ch_left, ch_right
        small_child, large_child = left, right
        small_metadata, large_metadata = left_metadata, right_metadata
        local_count = right_metadata.local_count
        partitioning = (
            maybe_remap_partitioning(
                ir,
                right_metadata.partitioning,
                child_ir=ir.children[1],
            )
            if ir.options[0] == "Right"
            else None
        )

    small_duplicated = small_metadata.duplicated
    need_allgather = comm.nranks > 1 and not small_duplicated
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
        comm,
        small_ch,
        small_child,
        need_allgather=need_allgather,
        collective_id=collective_id,
        ir_context=ir_context,
        concat_size_limit=(target_partition_size if ir.options[0] == "Inner" else None),
    )

    while (msg := await large_ch.recv(context)) is not None:
        await _broadcast_join_large_chunk(
            context,
            ir,
            ir_context,
            ch_out,
            small_dfs,
            small_child,
            TableChunk.from_message(msg, br=context.br()).make_available_and_spill(
                context.br(), allow_overbooking=True
            ),
            large_child,
            msg.sequence_number,
            small_size,
            broadcast_side,
            tracer=tracer,
        )

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
    if ir.options[0] == "Right":
        join_keys_for_output = ir.right_on
    else:
        join_keys_for_output = ir.left_on
    output_key_indices = names_to_indices(join_keys_for_output, ir.schema)
    return (
        left_key_indices[:n_keys],
        right_key_indices[:n_keys],
        output_key_indices[:n_keys],
    )


async def _join_chunks(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    tracer: ActorTracer | None,
) -> None:
    # Consume metadata from both shuffle outputs before reading data
    await gather_in_task_group(
        recv_metadata(ch_left, context),
        recv_metadata(ch_right, context),
    )

    left, right = ir.children
    while True:
        left_msg, right_msg = await gather_in_task_group(
            ch_left.recv(context), ch_right.recv(context)
        )
        if left_msg is None or right_msg is None:
            assert left_msg is None, (
                "Mismatched chunk counts in shuffle join: left has unmatched chunk. "
                f"Seq num: {left_msg.sequence_number}"
            )
            assert right_msg is None, (
                "Mismatched chunk counts in shuffle join: right has unmatched chunk. "
                f"Seq num: {right_msg.sequence_number}"
            )
            break
        assert left_msg.sequence_number == right_msg.sequence_number, (
            "Mismatched chunk sequence numbers in shuffle join. "
            f"Left: {left_msg.sequence_number}, Right: {right_msg.sequence_number}"
        )

        left_chunk = TableChunk.from_message(
            left_msg, br=context.br()
        ).make_available_and_spill(context.br(), allow_overbooking=True)
        right_chunk = TableChunk.from_message(
            right_msg, br=context.br()
        ).make_available_and_spill(context.br(), allow_overbooking=True)

        input_bytes = sum(
            col.device_buffer_size()
            for col in (
                *left_chunk.table_view().columns(),
                *right_chunk.table_view().columns(),
            )
        )
        with opaque_memory_usage(
            await reserve_memory(context, size=input_bytes, net_memory_delta=0)
        ):
            df = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                chunk_to_frame(left_chunk, left),
                chunk_to_frame(right_chunk, right),
                context=ir_context,
            )
            del left_chunk, right_chunk
        if tracer is not None:
            tracer.add_chunk(table=df.table)

        await ch_out.send(
            context,
            Message(
                left_msg.sequence_number,
                TableChunk.from_pylibcudf_table(
                    df.table, df.stream, exclusive_view=True, br=context.br()
                ),
            ),
        )
        del df

    await ch_out.drain(context)


def _log_shuffle_strategy_decision(
    tracer: ActorTracer,
    strategy: JoinStrategy,
    partitioning_left: NormalizedPartitioning,
    partitioning_right: NormalizedPartitioning,
) -> None:
    left_partitioning_desired = NormalizedPartitioning(
        inter_rank_modulus=strategy.shuffle_modulus,
        inter_rank_indices=strategy.left_indices,
        local_modulus=None,
        local_indices=(),
    )
    right_partitioning_desired = NormalizedPartitioning(
        inter_rank_modulus=strategy.shuffle_modulus,
        inter_rank_indices=strategy.right_indices,
        local_modulus=None,
        local_indices=(),
    )
    left_partitioned = partitioning_left == left_partitioning_desired
    right_partitioned = partitioning_right == right_partitioning_desired
    if left_partitioned and right_partitioned:
        tracer.decision = "chunkwise"
    elif left_partitioned:
        tracer.decision = "shuffle_right"
    elif right_partitioned:
        tracer.decision = "shuffle_left"
    else:
        tracer.decision = "shuffle"


async def _shuffle_join(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    strategy: JoinStrategy,
    collective_ids: list[int],
    *,
    tracer: ActorTracer | None,
) -> None:
    """Execute a shuffle (hash) join."""
    # Send output metadata
    shuffle_modulus = strategy.shuffle_modulus
    output_indices = strategy.output_indices
    nranks = comm.nranks
    metadata_out = ChannelMetadata(
        local_count=max(1, shuffle_modulus // nranks),
        partitioning=Partitioning(
            HashScheme(column_indices=output_indices, modulus=shuffle_modulus),
            local="inherit",
        )
        if output_indices
        else None,
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    # Construct a shuffle-shuffle-join pipeline.
    # The shuffle operations will pass chunks through unchanged
    # if the data is already partitioned correctly.
    ch_left_shuffle = context.create_channel()
    ch_right_shuffle = context.create_channel()
    # note: this is an actor inside of an actor. How should we log that in our traces?
    async with shutdown_on_error(
        context, ch_left_shuffle, ch_right_shuffle, trace_ir=ir, ir_context=ir_context
    ):
        actor_tasks = [
            _global_shuffle(
                context,
                comm,
                ir_context,
                ch_left_shuffle,
                ch_left,
                strategy.left_indices,
                strategy.shuffle_modulus,
                collective_ids.pop(0),
            ),
            _global_shuffle(
                context,
                comm,
                ir_context,
                ch_right_shuffle,
                ch_right,
                strategy.right_indices,
                strategy.shuffle_modulus,
                collective_ids.pop(0),
            ),
            _join_chunks(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left_shuffle,
                ch_right_shuffle,
                tracer=tracer,
            ),
        ]
        await gather_in_task_group(*actor_tasks)


def _make_shuffle_strategy(
    ir: Join,
    shuffle_modulus: int,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
) -> JoinStrategy:
    """Make a shuffle strategy."""
    # Use the coarsest prefix so we only shuffle on keys one side may already have
    n_left = len(left_partitioning.inter_rank_indices)
    n_right = len(right_partitioning.inter_rank_indices)
    if n_left and n_right:
        n_partitioned_keys = min(n_left, n_right)
    elif n_left or n_right:
        n_partitioned_keys = max(n_left, n_right)
    else:
        n_partitioned_keys = None  # both unpartitioned: shuffle on all join keys

    left_key_indices, right_key_indices, output_key_indices = _get_key_indices(
        ir, n_partitioned_keys
    )

    return JoinStrategy(
        shuffle_modulus=shuffle_modulus,
        output_indices=output_key_indices,
        left_indices=left_key_indices,
        right_indices=right_key_indices,
    )


async def _aggregate_estimates(
    context: Context,
    comm: Communicator,
    left_sample: JoinSideStats,
    right_sample: JoinSideStats,
    collective_ids: list[int],
) -> tuple[JoinSideStats, JoinSideStats]:
    """Aggregate table-size and row estimates across ranks."""
    # AllGather size, row, and chunk count estimates across ranks
    (
        left_total,
        right_total,
        left_total_rows,
        right_total_rows,
        left_total_chunks,
        right_total_chunks,
    ) = await allgather_reduce(
        context,
        comm,
        collective_ids.pop(),
        left_sample.total_size,
        right_sample.total_size,
        left_sample.total_rows,
        right_sample.total_rows,
        left_sample.total_chunks,
        right_sample.total_chunks,
    )

    new_left_sample = JoinSideStats(
        chunks=left_sample.chunks,
        total_size=left_total,
        total_rows=left_total_rows,
        total_chunks=left_total_chunks,
    )
    new_right_sample = JoinSideStats(
        chunks=right_sample.chunks,
        total_size=right_total,
        total_rows=right_total_rows,
        total_chunks=right_total_chunks,
    )
    return new_left_sample, new_right_sample


async def _choose_strategy_from_samples(
    comm: Communicator,
    ir: Join,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    executor: StreamingExecutor,
    *,
    left_sample: JoinSideStats,
    right_sample: JoinSideStats,
    chunkwise: bool,
    tracer: ActorTracer | None,
) -> JoinStrategy:
    """Choose potential broadcast side and minimum shuffle modulus."""
    if chunkwise:
        if tracer is not None:
            tracer.decision = "chunkwise"
        # TODO: Ensure this emits a "dynamic planning" decision of "chunkwise"
        # Or push it up a level to the caller?
        return _make_shuffle_strategy(
            ir,
            left_partitioning.inter_rank_modulus,
            left_partitioning,
            right_partitioning,
        )

    left_total, right_total = left_sample.total_size, right_sample.total_size
    left_total_rows, right_total_rows = left_sample.total_rows, right_sample.total_rows
    left_total_chunks, right_total_chunks = (
        left_sample.total_chunks,
        right_sample.total_chunks,
    )

    # =====================================================================
    # Broadcast-Join Strategy Selection
    # =====================================================================
    # - Inner: can broadcast either side
    # - Left/Semi/Anti: must broadcast right (stream left to preserve all left rows)
    # - Right: must broadcast left (stream right to preserve all right rows)
    # - Full: cannot broadcast (must shuffle both to preserve both sides)

    # Determine which sides may be broadcasted
    broadcast_threshold = executor.target_partition_size * executor.broadcast_join_limit
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
    if broadcast_side is not None:
        return JoinStrategy(broadcast_side=broadcast_side)

    # Couldn't broadcast - Use a shuffle join instead.
    estimated_output_size = max(left_total, right_total)
    ideal_output_count = max(1, estimated_output_size // executor.target_partition_size)
    # Limit the output count to 10x the larger input side.
    # This is an arbitrary limit to prevent an oversized sample
    # from blowing up the chunk count.
    max_output_chunks = 10 * max(left_total_chunks, right_total_chunks)
    min_shuffle_modulus = min(ideal_output_count, max_output_chunks)

    # Stay away from cuDF's row limit
    if (estimated_rows_count := max(left_total_rows, right_total_rows)) > 0:
        max_rows_per_partition = CUDF_ROW_LIMIT // 4
        min_partitions_for_row_limit = (
            estimated_rows_count + max_rows_per_partition - 1
        ) // max_rows_per_partition
        min_shuffle_modulus = max(min_shuffle_modulus, min_partitions_for_row_limit)

    shuffle_modulus = _choose_shuffle_modulus(
        comm,
        left_partitioning,
        right_partitioning,
        min_shuffle_modulus,
    )  # Global modulus

    strategy = _make_shuffle_strategy(
        ir, shuffle_modulus, left_partitioning, right_partitioning
    )

    if tracer is not None:
        _log_shuffle_strategy_decision(
            tracer,
            strategy,
            left_partitioning,
            right_partitioning,
        )
    return strategy


def _choose_shuffle_modulus(
    comm: Communicator,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    min_shuffle_modulus: int,
) -> int:
    """Choose an appropriate modulus for a shuffle join."""
    left_modulus = (
        left_partitioning.inter_rank_modulus if left_partitioning is not None else None
    )
    right_modulus = (
        right_partitioning.inter_rank_modulus
        if right_partitioning is not None
        else None
    )
    default_modulus = max(comm.nranks, min_shuffle_modulus)
    small, large = sorted(
        [left_modulus or default_modulus, right_modulus or default_modulus]
    )
    if large % small == 0 and small >= min_shuffle_modulus:
        return small
    else:
        return max(large, min_shuffle_modulus)


async def _sample_chunks(
    context: Context,
    ch: Channel[TableChunk],
    max_sample_chunks: int,
    max_sample_bytes: int,
    local_count: int,
) -> JoinSideStats:
    """
    Sample chunks from a channel.

    Parameters
    ----------
    context
        The context.
    ch
        The channel to sample from.
    max_sample_chunks
        The maximum number of chunks to sample.
    max_sample_bytes
        The maximum number of bytes to sample.
    local_count
        The number of local chunks.

    Returns
    -------
    The sampled chunks.
    """
    sampled_chunks: dict[int, TableChunk] = {}
    total_size = 0
    total_rows = 0
    for _ in range(max_sample_chunks):
        msg = await ch.recv(context)
        if msg is None:
            break
        chunk = TableChunk.from_message(msg, br=context.br()).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        sampled_chunks[msg.sequence_number] = chunk
        total_size += chunk.data_alloc_size()
        total_rows += chunk.shape[0]
        if total_size >= max_sample_bytes:
            break
    if sampled_chunks:
        total_size = int((total_size / len(sampled_chunks)) * local_count)
        total_rows = int((total_rows / len(sampled_chunks)) * local_count)
    return JoinSideStats(
        chunks=sampled_chunks,
        total_size=total_size,
        total_rows=total_rows,
        total_chunks=local_count,
    )


async def _choose_strategy(
    context: Context,
    comm: Communicator,
    ir: Join,
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    executor: StreamingExecutor,
    collective_ids: list[int],
    *,
    tracer: ActorTracer | None,
) -> tuple[JoinSideStats, JoinSideStats, JoinStrategy]:
    """Sample both sides, aggregate estimates, and choose broadcast vs shuffle."""
    nranks = comm.nranks
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

    if left_partitioning.is_compatible_with(right_partitioning):
        # We can use a chunkwise join
        chunkwise = True
        left_sample = JoinSideStats(total_chunks=left_metadata.local_count)
        right_sample = JoinSideStats(total_chunks=right_metadata.local_count)
    else:
        # Need to shuffle or broadcast - Use sampled data to choose a strategy
        chunkwise = False
        assert executor.dynamic_planning is not None
        sample_chunk_count = executor.dynamic_planning.sample_chunk_count
        target_partition_size = executor.target_partition_size
        left_sample, right_sample = await gather_in_task_group(
            _sample_chunks(
                context,
                ch_left,
                sample_chunk_count,
                target_partition_size,
                left_metadata.local_count,
            ),
            _sample_chunks(
                context,
                ch_right,
                sample_chunk_count,
                target_partition_size,
                right_metadata.local_count,
            ),
        )
        left_sample, right_sample = await _aggregate_estimates(
            context,
            comm,
            left_sample,
            right_sample,
            collective_ids,
        )

    strategy = await _choose_strategy_from_samples(
        comm,
        ir,
        left_metadata,
        right_metadata,
        left_partitioning,
        right_partitioning,
        executor,
        left_sample=left_sample,
        right_sample=right_sample,
        chunkwise=chunkwise,
        tracer=tracer,
    )

    return left_sample, right_sample, strategy


def _should_use_bloom_filter(
    ir: Join,
    strategy: JoinStrategy,
    left_sample: JoinSideStats,
    right_sample: JoinSideStats,
    threshold: float,
) -> bool:
    """Return True if bloom filter pre-filtering should be applied."""
    if threshold == 0.0 or strategy.shuffle_modulus == 0:
        return False
    if ir.options[0] not in ("Inner", "Semi"):
        return False
    large_rows = max(left_sample.total_rows, right_sample.total_rows)
    small_rows = min(left_sample.total_rows, right_sample.total_rows)
    return large_rows > 0 and small_rows / large_rows < threshold


async def _replay_with_metadata(
    context: Context,
    ch_out: Channel[TableChunk],
    metadata: ChannelMetadata,
    messages: list[Message],
    trace_ir: Join,
) -> None:
    """Send metadata then all messages to ch_out, then drain."""
    async with shutdown_on_error(context, ch_out, trace_ir=trace_ir):
        await send_metadata(ch_out, context, metadata)
        for msg in messages:
            await ch_out.send(context, msg)
        await ch_out.drain(context)


async def _relay_chunks_raw(
    context: Context,
    ch_out: Channel[TableChunk],
    buffered_chunks: dict[int, TableChunk],
    ch_in: Channel[TableChunk],
    trace_ir: Join,
) -> None:
    """Relay buffered chunks then remaining channel to ch_out, without metadata."""
    async with shutdown_on_error(context, ch_out, ch_in, trace_ir=trace_ir):
        for seq, chunk in sorted(buffered_chunks.items()):
            await ch_out.send(context, Message(seq, chunk))
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
        await ch_out.drain(context)


async def _forward_with_metadata(
    context: Context,
    ch_out: Channel[TableChunk],
    metadata: ChannelMetadata,
    ch_in: Channel[TableChunk],
    trace_ir: Join,
) -> None:
    """Send metadata to ch_out then forward all messages from ch_in."""
    async with shutdown_on_error(context, ch_out, ch_in, trace_ir=trace_ir):
        await send_metadata(ch_out, context, metadata)
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
        await ch_out.drain(context)


async def _bloom_shuffle_join(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_sample: JoinSideStats,
    right_sample: JoinSideStats,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    strategy: JoinStrategy,
    collective_ids: list[int],
    *,
    tracer: ActorTracer | None,
) -> None:
    """Shuffle join with bloom filter pre-filtering on the large side."""
    # Determine small/large sides by row count
    if left_sample.total_rows <= right_sample.total_rows:
        small_chunks_init, large_chunks_init = left_sample.chunks, right_sample.chunks
        small_metadata, large_metadata = left_metadata, right_metadata
        ch_small, ch_large = ch_left, ch_right
        small_key_indices, large_key_indices = (
            strategy.left_indices,
            strategy.right_indices,
        )
        small_is_left = True
    else:
        small_chunks_init, large_chunks_init = right_sample.chunks, left_sample.chunks
        small_metadata, large_metadata = right_metadata, left_metadata
        ch_small, ch_large = ch_right, ch_left
        small_key_indices, large_key_indices = (
            strategy.right_indices,
            strategy.left_indices,
        )
        small_is_left = False

    # Drain the small side into a spillable buffer while simultaneously
    # extracting key-only chunks for bloom build.
    small_buffer = context.spillable_messages()
    small_mids: list[tuple[int, int]] = []  # (seq_num, mid) in insertion order
    small_key_msgs: list[Message] = []

    def _make_key_msg(chunk: TableChunk, seq: int) -> Message:
        tv = chunk.table_view()
        int32 = plc.DataType(plc.TypeId.INT32)
        init = plc.Scalar.from_py(0, dtype=int32, stream=chunk.stream)
        step = plc.Scalar.from_py(1, dtype=int32, stream=chunk.stream)
        gather_map = plc.filling.sequence(
            tv.num_rows(), init, step, stream=chunk.stream
        )
        key_table = plc.copying.gather(
            plc.Table([tv.columns()[i] for i in small_key_indices]),
            gather_map,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            stream=chunk.stream,
        )
        return Message(
            seq,
            TableChunk.from_pylibcudf_table(
                key_table, chunk.stream, exclusive_view=True
            ),
        )

    for seq, stale_chunk in sorted(small_chunks_init.items()):
        chunk = stale_chunk.make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        small_key_msgs.append(_make_key_msg(chunk, seq))
        small_mids.append((seq, small_buffer.insert(Message(seq, chunk))))

    while (msg := await ch_small.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        seq = msg.sequence_number
        small_key_msgs.append(_make_key_msg(chunk, seq))
        small_mids.append((seq, small_buffer.insert(Message(seq, chunk))))

    spill_func_id = context.br().spill_manager.add_spill_function(
        make_spill_function([small_buffer], context), priority=0
    )
    try:
        bloom = BloomFilter(
            context,
            comm,
            seed=0,
            num_filter_blocks=BloomFilter.fitting_num_blocks(_BLOOM_L2_SIZE),
        )
        bloom_tag = collective_ids.pop(-1)
        ch_build_in = context.create_channel()
        ch_filter = context.create_channel()
        pull_filter_actor, deferred_filter = pull_from_channel(context, ch_filter)
        async with shutdown_on_error(
            context, ch_build_in, ch_filter, trace_ir=ir, ir_context=ir_context
        ):
            await asyncio.to_thread(
                run_actor_network,
                actors=[
                    push_to_channel(context, ch_build_in, small_key_msgs),
                    bloom.build(ch_build_in, ch_filter, tag=bloom_tag),
                    pull_filter_actor,
                ],
            )
            (filter_message,) = deferred_filter.release()

        # Stream the large side through bloom.apply without buffering it.
        # cpp_set_py_future uses asyncio.run_coroutine_threadsafe, so the Python
        # coroutines on the event loop and bloom.apply in the thread pool share
        # channels safely.
        small_replay_msgs = [
            small_buffer.extract(mid=mid) for _, mid in sorted(small_mids)
        ]
        ch_filter_for_apply = context.create_channel()
        ch_large_raw = context.create_channel()
        ch_large_filtered = context.create_channel()
        ch_small_replay = context.create_channel()
        ch_large_replay = context.create_channel()
        ch_left_new = ch_small_replay if small_is_left else ch_large_replay
        ch_right_new = ch_large_replay if small_is_left else ch_small_replay

        async with shutdown_on_error(
            context,
            ch_filter_for_apply,
            ch_large_raw,
            ch_large_filtered,
            ch_small_replay,
            ch_large_replay,
            trace_ir=ir,
            ir_context=ir_context,
        ) as tracer:
            await asyncio.gather(
                _replay_with_metadata(
                    context,
                    ch_small_replay,
                    small_metadata,
                    small_replay_msgs,
                    trace_ir=ir,
                ),
                _relay_chunks_raw(
                    context, ch_large_raw, large_chunks_init, ch_large, trace_ir=ir
                ),
                asyncio.to_thread(
                    run_actor_network,
                    actors=[
                        push_to_channel(context, ch_filter_for_apply, [filter_message]),
                        bloom.apply(
                            ch_filter_for_apply,
                            ch_large_raw,
                            ch_large_filtered,
                            keys=large_key_indices,
                        ),
                    ],
                ),
                _forward_with_metadata(
                    context,
                    ch_large_replay,
                    large_metadata,
                    ch_large_filtered,
                    trace_ir=ir,
                ),
                _shuffle_join(
                    context,
                    comm,
                    ir,
                    ir_context,
                    ch_out,
                    ch_left_new,
                    ch_right_new,
                    strategy,
                    collective_ids,
                    tracer=tracer,
                ),
            )
    finally:
        context.br().spill_manager.remove_spill_function(spill_func_id)


@define_actor()
async def join_actor(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    executor: StreamingExecutor,
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
    comm
        The communicator.
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
    executor
        Streaming executor configuration.
    collective_ids
        List of collective IDs for shuffle/broadcast; consumed as needed.
    """
    async with shutdown_on_error(
        context,
        ch_out,
        ch_left,
        ch_right,
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        left_metadata, right_metadata = await gather_in_task_group(
            recv_metadata(ch_left, context),
            recv_metadata(ch_right, context),
        )

        left_sample, right_sample, strategy = await _choose_strategy(
            context,
            comm,
            ir,
            ch_left,
            ch_right,
            left_metadata,
            right_metadata,
            executor,
            collective_ids,
            tracer=tracer,
        )

        bloom_threshold = (
            executor.dynamic_planning.bloom_filter_threshold
            if executor.dynamic_planning is not None
            else 0.0
        )
        if strategy.broadcast_side is None and _should_use_bloom_filter(
            ir, strategy, left_sample, right_sample, bloom_threshold
        ):
            if tracer is not None:
                tracer.decision = f"{tracer.decision}_filtered"
            await _bloom_shuffle_join(
                context,
                comm,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                left_sample,
                right_sample,
                left_metadata,
                right_metadata,
                strategy,
                collective_ids,
                tracer=tracer,
            )
            return

        ch_left_replay = context.create_channel()
        ch_right_replay = context.create_channel()
        async with shutdown_on_error(
            context,
            ch_left_replay,
            ch_right_replay,
            trace_ir=ir,
            ir_context=ir_context,
        ):
            actor_tasks = [
                replay_buffered_channel(
                    context,
                    ch_left_replay,
                    ch_left,
                    left_sample.chunks,
                    left_metadata,
                    trace_ir=ir,
                ),
                replay_buffered_channel(
                    context,
                    ch_right_replay,
                    ch_right,
                    right_sample.chunks,
                    right_metadata,
                    trace_ir=ir,
                ),
            ]
            ch_left = ch_left_replay
            ch_right = ch_right_replay

            if strategy.broadcast_side is not None:
                actor_tasks.append(
                    _broadcast_join(
                        context,
                        comm,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        strategy,
                        collective_ids,
                        executor.target_partition_size,
                        tracer=tracer,
                    )
                )
            else:
                actor_tasks.append(
                    _shuffle_join(
                        context,
                        comm,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        strategy,
                        collective_ids,
                        tracer=tracer,
                    )
                )
            await gather_in_task_group(*actor_tasks)


def _use_pwise_join(
    executor: StreamingExecutor,
    partition_info: MutableMapping[IR, PartitionInfo],
    ir: Join,
) -> bool:
    """Whether to use a static-planning partition-wise join."""
    left, right = ir.children
    output_count = partition_info[ir].count
    if (
        output_count == 1
        and isinstance(left, Repartition)
        and isinstance(right, Repartition)
    ):
        # We fell back to single-partition behavior at lowering time
        return True

    if executor.name == "streaming" and executor.dynamic_planning is not None:
        return False

    left_count = partition_info[left].count
    right_count = partition_info[right].count
    left_partitioned = (
        partition_info[left].partitioned_on == ir.left_on and left_count == output_count
    )
    right_partitioned = (
        partition_info[right].partitioned_on == ir.right_on
        and right_count == output_count
    )
    return left_partitioned and right_partitioned


@generate_ir_sub_network.register(Join)
def _(
    ir: Join, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Join operation.
    left, right = ir.children
    partition_info = rec.state["partition_info"]
    left_count = partition_info[left].count
    right_count = partition_info[right].count
    executor = rec.state["config_options"].executor
    pwise_join = _use_pwise_join(executor, partition_info, ir)

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

    elif (
        executor.name == "streaming"
        and executor.dynamic_planning is not None
        and ir.options[0] in ("Inner", "Left", "Right", "Full", "Semi", "Anti")
    ):
        # Dynamic join - decide strategy at runtime
        collective_ids = list(rec.state["collective_id_map"].get(ir, []))
        # Join uses up to 4 collective IDs: left shuffle, right shuffle, bloom filter, allgather
        if len(collective_ids) < 4:
            raise ValueError(
                "Dynamic join requires 4 reserved collective IDs "
                "(left shuffle + right shuffle + bloom filter + allgather); got "
                f"{len(collective_ids)} for this Join. "
                "Ensure ReserveOpIDs is run with dynamic_planning enabled."
            )
        actors[ir] = [
            join_actor(
                rec.state["context"],
                rec.state["comm"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                executor,
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
                rec.state["comm"],
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
