# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic Distinct (Unique) node for rapidsmpf runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.communicator.single import new_communicator as single_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

from cudf_polars.dsl.ir import IR, Distinct
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    allgather_reduce,
    chunkwise_evaluate,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    is_partitioned_on_keys,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.utils.config import StreamingExecutor

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel

    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer


async def _tree_distinct(
    context: Context,
    ir: Distinct,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    target_partition_size: int,
    *,
    evaluated_chunks: list[TableChunk] | None = None,
    collective_id: int | None = None,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Tree-based distinct reduction.

    Reads chunks from input, applies distinct, and reduces incrementally.
    When collective_id is provided, uses allgather to collect partial
    results from all ranks before final distinct.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    ir
        The Distinct IR node.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    metadata_in
        The input channel metadata.
    target_partition_size
        Target size in bytes for output partitions.
    evaluated_chunks
        Chunks that have already been evaluated (e.g., during sampling).
    collective_id
        Optional collective ID for allgather. If None, no allgather is performed.
    tracer
        Optional tracer for runtime metrics.
    """
    metadata_out = ChannelMetadata(
        local_count=1,
        partitioning=None,
        duplicated=True if collective_id is not None else metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    distinct_chunks: list[TableChunk] = list(evaluated_chunks or [])
    total_size = sum(c.data_alloc_size() for c in distinct_chunks)

    receiving = True
    while receiving or len(distinct_chunks) > 1:
        if receiving:
            msg = await ch_in.recv(context)
            if msg is None:
                receiving = False
            else:
                chunk = await evaluate_chunk(
                    context, TableChunk.from_message(msg), ir, ir_context
                )
                del msg
                distinct_chunks.append(chunk)
                total_size += chunk.data_alloc_size()

        if len(distinct_chunks) > 1 and (
            not receiving or total_size > target_partition_size
        ):
            merged = await evaluate_batch(distinct_chunks, context, ir, ir_context)
            distinct_chunks = [merged]
            total_size = merged.data_alloc_size()

    if collective_id is not None:
        allgather = AllGatherManager(context, collective_id)
        stream = ir_context.get_cuda_stream()

        if distinct_chunks:
            allgather.insert(0, distinct_chunks[0])
        allgather.insert_finished()

        gathered_chunk = TableChunk.from_pylibcudf_table(
            await allgather.extract_concatenated(stream),
            stream,
            exclusive_view=True,
        )
        distinct_chunks = [
            await evaluate_chunk(context, gathered_chunk, ir, ir_context)
        ]
        del gathered_chunk

    # Send final result
    if distinct_chunks:
        if tracer is not None:
            tracer.add_chunk(table=distinct_chunks[0].table_view())
        await ch_out.send(context, Message(0, distinct_chunks[0]))
    else:
        stream = ir_context.get_cuda_stream()
        empty_chunk = empty_table_chunk(ir, context, stream)
        if tracer is not None:
            tracer.add_chunk(table=empty_chunk.table_view())
        await ch_out.send(context, Message(0, empty_chunk))

    await ch_out.drain(context)


async def _shuffle_distinct(
    context: Context,
    ir: Distinct,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    output_count: int,
    collective_id: int,
    key_indices: tuple[int, ...],
    *,
    evaluated_chunks: list[TableChunk] | None = None,
    shuffle_context: Context | None = None,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Shuffle-based distinct.

    Shuffles data by distinct keys, then applies local distinct to
    each partition. Use when the expected output is large.

    Parameters
    ----------
    context
        The rapidsmpf streaming context for channel operations.
    ir
        The Distinct IR node.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    metadata_in
        The input channel metadata.
    output_count
        The number of output partitions per rank.
    collective_id
        The collective ID for the shuffle operation.
    key_indices
        The column indices of the distinct keys.
    evaluated_chunks
        Chunks that have already been evaluated (e.g., during sampling).
    shuffle_context
        Optional context for shuffle operations.
        Defaults to a temporary local context.
    tracer
        Optional tracer for runtime metrics.
    """
    # Define shuffle context
    if shuffle_context is None:
        # Create a temporary local context
        options = Options(get_environment_variables())
        local_comm = single_comm(options)
        shuffle_context = Context(local_comm, context.br(), options)
    shuf_nranks = shuffle_context.comm().nranks
    shuf_rank = shuffle_context.comm().rank

    # Calculate output partitioning
    modulus = shuf_nranks * output_count

    # Send output metadata
    # For local shuffle (shuf_nranks=1), keep inter-rank partitioning from input
    if shuf_nranks == 1 and metadata_in.partitioning is not None:
        # Local shuffle: preserve inter-rank partitioning
        inter_rank_scheme = metadata_in.partitioning.inter_rank
        local_scheme = HashScheme(column_indices=key_indices, modulus=modulus)
    else:
        # Global shuffle: use hash partitioning
        inter_rank_scheme = HashScheme(column_indices=key_indices, modulus=modulus)
        local_scheme = "inherit"

    metadata_out = ChannelMetadata(
        local_count=output_count,
        partitioning=Partitioning(inter_rank_scheme, local_scheme),
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    # Create shuffle manager with shuffle context
    shuffle = ShuffleManager(shuffle_context, output_count, key_indices, collective_id)

    for chunk in evaluated_chunks or []:
        shuffle.insert_chunk(
            chunk.make_available_and_spill(shuffle_context.br(), allow_overbooking=True)
        )

    # Apply distinct to remaining chunks before inserting into shuffler
    while (msg := await ch_in.recv(context)) is not None:
        distinct_chunk = await evaluate_chunk(
            context, TableChunk.from_message(msg), ir, ir_context
        )
        del msg
        shuffle.insert_chunk(
            distinct_chunk.make_available_and_spill(
                shuffle_context.br(), allow_overbooking=True
            )
        )

    await shuffle.insert_finished()

    stream = ir_context.get_cuda_stream()
    for seq_num, partition_id in enumerate(range(shuf_rank, output_count, shuf_nranks)):
        partition_chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
        )
        output_chunk = await evaluate_chunk(context, partition_chunk, ir, ir_context)
        del partition_chunk
        if tracer is not None:
            tracer.add_chunk(table=output_chunk.table_view())
        await ch_out.send(context, Message(seq_num, output_chunk))

    await ch_out.drain(context)


@define_py_node()
async def distinct_node(
    context: Context,
    ir: Distinct,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    sample_chunk_count: int,
    target_partition_size: int,
    collective_ids: list[int],
) -> None:
    """
    Dynamic Distinct node that selects the best strategy at runtime.

    Strategy selection based on partitioning and sampled data:

    - Chunkwise: Data fully partitioned (inter-rank and local) - apply
      distinct per chunk with no reduction.
    - Tree local: Partitioned inter-rank, small output - local tree
      reduction, no global communication.
    - Shuffle local: Partitioned inter-rank, large output - local hash
      shuffle with single-rank communicator.
    - Tree allgather: Small estimated output - tree reduction with
      allgather to merge across ranks.
    - Shuffle: Large estimated output requiring global redistribution.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    ir
        The Distinct IR node.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    sample_chunk_count
        Number of chunks to sample for size estimation.
    target_partition_size
        Target size (in bytes) for output partitions.
    collective_ids
        Pool of collective IDs for allgather and shuffle operations.
    """
    async with shutdown_on_error(context, ch_in, ch_out, trace_ir=ir) as tracer:
        # Receive input metadata
        metadata_in = await recv_metadata(ch_in, context)

        # Get distinct key column indices
        input_schema_keys = list(ir.children[0].schema.keys())
        subset = ir.subset or frozenset(ir.schema)
        key_indices = tuple(
            input_schema_keys.index(k) for k in subset if k in input_schema_keys
        )

        # Check if already partitioned on keys
        # - partitioned_inter_rank: data is partitioned between ranks
        # - partitioned_local: data is also partitioned within rank (per chunk)
        nranks = context.comm().nranks
        partitioned_inter_rank, partitioned_local = is_partitioned_on_keys(
            metadata_in,
            key_indices,
            nranks,
        )

        # Determine if we can skip global communication
        can_skip_global_comm = metadata_in.duplicated or partitioned_inter_rank

        # If both inter-rank and local are partitioned, each chunk has
        # disjoint keys - we can do simple chunkwise distinct
        fully_partitioned = partitioned_inter_rank and partitioned_local

        # Detect if lowering already collapsed input to single partition
        # (e.g., for KEEP_NONE with ordering, complex slices)
        fallback_case = (
            metadata_in.local_count == 1
            and (metadata_in.duplicated or nranks == 1)
            and isinstance(ir.children[0], Repartition)
        )

        # If already fully partitioned or concatenated, use chunkwise evaluation
        if fully_partitioned or fallback_case:
            await chunkwise_evaluate(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                tracer=tracer,
            )
            return

        require_tree = ir.stable or ir.keep in (
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
        )

        evaluated_chunks: list[TableChunk] = []
        total_size = 0
        merge_count = 0
        chunks_sampled = 0

        for _ in range(sample_chunk_count):
            msg = await ch_in.recv(context)
            if msg is None:
                break
            chunks_sampled += 1
            chunk = await evaluate_chunk(
                context, TableChunk.from_message(msg), ir, ir_context
            )
            del msg
            total_size += chunk.data_alloc_size(MemoryType.DEVICE)
            evaluated_chunks.append(chunk)

            if total_size > target_partition_size and len(evaluated_chunks) > 1:
                merged = await evaluate_batch(evaluated_chunks, context, ir, ir_context)
                total_size = merged.data_alloc_size(MemoryType.DEVICE)
                evaluated_chunks = [merged]
                merge_count += 1
                if total_size > target_partition_size:
                    break

        local_count = metadata_in.local_count
        if can_skip_global_comm:
            global_chunk_count = local_count
            global_chunks_sampled = chunks_sampled
        else:
            (
                total_size,
                global_chunk_count,
                global_chunks_sampled,
            ) = await allgather_reduce(
                context, collective_ids.pop(), total_size, local_count, chunks_sampled
            )

        if global_chunks_sampled > 0:
            global_size = (total_size // global_chunks_sampled) * global_chunk_count
        else:
            global_size = 0

        # TODO: Fast return if we already have the slice ready
        if ir.zlice is not None and ir.zlice[1] is not None and evaluated_chunks:
            total_rows = sum(c.table_view().num_rows() for c in evaluated_chunks)
            if total_rows > 0:
                avg_row_size = total_size / total_rows
                global_size = min(global_size, int(avg_row_size * ir.zlice[1]))

        use_tree = global_size < target_partition_size or require_tree

        if can_skip_global_comm:
            if use_tree:
                if tracer is not None:
                    tracer.decision = "tree_local"
                await _tree_distinct(
                    context,
                    ir,
                    ir_context,
                    ch_out,
                    ch_in,
                    metadata_in,
                    target_partition_size,
                    evaluated_chunks=evaluated_chunks,
                    tracer=tracer,
                )
            else:
                if tracer is not None:
                    tracer.decision = "shuffle_local"
                ideal_count = max(1, global_size // target_partition_size)
                output_count = max(1, min(ideal_count, local_count))
                await _shuffle_distinct(
                    context,
                    ir,
                    ir_context,
                    ch_out,
                    ch_in,
                    metadata_in,
                    output_count,
                    collective_ids.pop(),
                    key_indices,
                    evaluated_chunks=evaluated_chunks,
                    shuffle_context=context if nranks == 1 else None,
                    tracer=tracer,
                )
        elif use_tree:
            if tracer is not None:
                tracer.decision = "tree_allgather"
            await _tree_distinct(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                target_partition_size,
                evaluated_chunks=evaluated_chunks,
                collective_id=collective_ids.pop(),
                tracer=tracer,
            )
        else:
            if tracer is not None:
                tracer.decision = "shuffle"
            ideal_count = max(1, global_size // target_partition_size)
            output_count = max(nranks, min(ideal_count, global_chunk_count))
            await _shuffle_distinct(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                output_count,
                collective_ids.pop(),
                key_indices,
                evaluated_chunks=evaluated_chunks,
                shuffle_context=context,
                tracer=tracer,
            )


@generate_ir_sub_network.register(Distinct)
def _(
    ir: Distinct, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]
    executor = config_options.executor

    if not isinstance(executor, StreamingExecutor) or executor.dynamic_planning is None:
        # Fall back to the default IR handler (bypass Distinct dispatch)
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    nodes, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    assert len(collective_ids) == 2, (
        f"Distinct requires 2 collective IDs, got {len(collective_ids)}"
    )
    nodes[ir] = [
        distinct_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            executor.dynamic_planning.sample_chunk_count_distinct,
            executor.target_partition_size,
            collective_ids,
        )
    ]
    return nodes, channels
