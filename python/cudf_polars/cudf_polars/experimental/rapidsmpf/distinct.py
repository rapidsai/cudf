# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic Distinct (Unique) node for rapidsmpf runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rapidsmpf.communicator.single import new_communicator as single_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
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
from cudf_polars.dsl.ir import IR, Distinct
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    allgather_reduce,
    empty_table_chunk,
    is_partitioned_on_keys,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.experimental.utils import _concat
from cudf_polars.utils.config import StreamingExecutor

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel

    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer


def _apply_distinct(
    chunk: TableChunk,
    ir: Distinct,
    ir_context: Any,
) -> TableChunk:
    """
    Apply Distinct evaluation to a chunk.

    Parameters
    ----------
    chunk
        The input table chunk.
    ir
        The Distinct IR node.
    ir_context
        The IR execution context.

    Returns
    -------
    The chunk with duplicates removed.
    """
    input_schema = ir.children[0].schema
    names = list(input_schema.keys())
    dtypes = list(input_schema.values())
    df = ir.do_evaluate(
        *ir._non_child_args,
        DataFrame.from_table(chunk.table_view(), names, dtypes, chunk.stream),
        context=ir_context,
    )
    return TableChunk.from_pylibcudf_table(df.table, chunk.stream, exclusive_view=True)


# ============================================================================
# Distinct Strategies
# ============================================================================


async def _tree_distinct(
    context: Context,
    ir: Distinct,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    initial_chunks: list[TableChunk],
    n_ary: int,
    *,
    collective_id: int | None = None,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Tree-based distinct reduction.

    Collects all chunks, applies local distinct, then performs k-ary tree
    reduction by concatenating and re-applying distinct until a single
    chunk remains.

    When collective_id is provided and data is not duplicated, uses allgather
    to collect partial results from all ranks before final distinct.

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
    initial_chunks
        Chunks already received during sampling.
    n_ary
        The fan-in for tree reduction.
    collective_id
        Optional collective ID for allgather. If None, no allgather is performed.
    tracer
        Optional tracer for runtime metrics.
    """
    nranks = context.comm().nranks
    need_allgather = (
        collective_id is not None and not metadata_in.duplicated and nranks > 1
    )

    # Output: single chunk, duplicated if allgather is used
    metadata_out = ChannelMetadata(
        local_count=1,
        partitioning=None,
        duplicated=True if need_allgather else metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    # Collect all chunks, applying local distinct
    distinct_chunks: list[TableChunk] = list(initial_chunks)

    # Apply distinct to remaining chunks from input channel
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg)
        del msg
        chunk, extra = await make_table_chunks_available_or_wait(
            context,
            chunk,
            reserve_extra=chunk.data_alloc_size(),
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            distinct_chunk = await asyncio.to_thread(
                _apply_distinct, chunk, ir, ir_context
            )
            distinct_chunks.append(distinct_chunk)
            del chunk

    # Tree reduction
    k = n_ary
    input_schema = ir.children[0].schema
    while len(distinct_chunks) > 1:
        new_chunks: list[TableChunk] = []
        for i in range(0, len(distinct_chunks), k):
            batch = distinct_chunks[i : i + k]
            if len(batch) == 1:
                new_chunks.append(batch[0])
            else:
                batch, extra = await make_table_chunks_available_or_wait(
                    context,
                    batch,
                    reserve_extra=sum(c.data_alloc_size() for c in batch),
                    net_memory_delta=0,
                )
                with opaque_memory_usage(extra):
                    concatenated = await asyncio.to_thread(
                        _concat,
                        *[
                            DataFrame.from_table(
                                c.table_view(),
                                list(input_schema.keys()),
                                list(input_schema.values()),
                                c.stream,
                            )
                            for c in batch
                        ],
                        context=ir_context,
                    )
                    df = await asyncio.to_thread(
                        ir.do_evaluate,
                        *ir._non_child_args,
                        concatenated,
                        context=ir_context,
                    )
                    del concatenated
                    new_chunks.append(
                        TableChunk.from_pylibcudf_table(
                            df.table, df.stream, exclusive_view=True
                        )
                    )
                    del df
        distinct_chunks = new_chunks

    # Allgather partial results from all ranks if needed
    if need_allgather:
        assert collective_id is not None

        allgather = AllGatherManager(context, collective_id)
        stream = ir_context.get_cuda_stream()

        # Insert the local distinct result (or empty if no local data)
        if distinct_chunks:
            chunk = distinct_chunks[0]
            allgather.insert(0, chunk)
        # else: No local data - don't insert anything into allgather
        # Empty table chunks can have schema mismatches (e.g., STRING columns
        # with 0 children vs 1 child), so we skip them entirely
        allgather.insert_finished()

        # Extract concatenated results from all ranks
        gathered_table = await allgather.extract_concatenated(stream)
        distinct_chunks = [
            TableChunk.from_pylibcudf_table(gathered_table, stream, exclusive_view=True)
        ]

        # One more distinct round to merge results from all ranks
        chunk = distinct_chunks[0]
        chunk, extra = await make_table_chunks_available_or_wait(
            context,
            chunk,
            reserve_extra=chunk.data_alloc_size(),
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            df = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                DataFrame.from_table(
                    chunk.table_view(),
                    list(input_schema.keys()),
                    list(input_schema.values()),
                    chunk.stream,
                ),
                context=ir_context,
            )
            output_chunk = TableChunk.from_pylibcudf_table(
                df.table, df.stream, exclusive_view=True
            )
            del df, chunk
        distinct_chunks = [output_chunk]

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
    initial_chunks: list[TableChunk],
    output_count: int,
    collective_id: int,
    key_indices: tuple[int, ...],
    *,
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
    initial_chunks
        Chunks already received during sampling.
    output_count
        The number of output partitions per rank.
    collective_id
        The collective ID for the shuffle operation.
    key_indices
        The column indices of the distinct keys.
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

    # Insert initial chunks
    for chunk in initial_chunks:
        shuffle.insert_chunk(
            chunk.make_available_and_spill(shuffle_context.br(), allow_overbooking=True)
        )

    # Insert remaining chunks from channel
    while (msg := await ch_in.recv(context)) is not None:
        shuffle.insert_chunk(
            TableChunk.from_message(msg).make_available_and_spill(
                shuffle_context.br(), allow_overbooking=True
            )
        )
        del msg

    await shuffle.insert_finished()

    # Extract shuffled partitions and apply local distinct
    input_schema = ir.children[0].schema
    stream = ir_context.get_cuda_stream()
    for seq_num, partition_id in enumerate(range(shuf_rank, output_count, shuf_nranks)):
        partition_chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
        )

        partition_chunk, extra = await make_table_chunks_available_or_wait(
            context,
            partition_chunk,
            reserve_extra=partition_chunk.data_alloc_size(),
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            df = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                DataFrame.from_table(
                    partition_chunk.table_view(),
                    list(input_schema.keys()),
                    list(input_schema.values()),
                    partition_chunk.stream,
                ),
                context=ir_context,
            )
            output_chunk = TableChunk.from_pylibcudf_table(
                df.table, df.stream, exclusive_view=True
            )
            if tracer is not None:
                tracer.add_chunk(table=output_chunk.table_view())
            del df, partition_chunk

        await ch_out.send(context, Message(seq_num, output_chunk))

    await ch_out.drain(context)


async def _chunkwise_distinct(
    context: Context,
    ir: Distinct,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    initial_chunks: list[TableChunk],
    *,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Chunkwise distinct - apply distinct to each chunk independently.

    Use when data is fully partitioned on the distinct keys (both inter-rank
    and locally), meaning each chunk has a disjoint set of key values.

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
    initial_chunks
        Chunks already received during sampling.
    tracer
        Optional tracer for runtime metrics.
    """
    # Output: preserve input metadata (partitioning unchanged)
    await send_metadata(ch_out, context, metadata_in)

    seq_num = 0

    # Process initial chunks (already distinct from sampling)
    for chunk in initial_chunks:
        if tracer is not None:
            tracer.add_chunk(table=chunk.table_view())
        await ch_out.send(context, Message(seq_num, chunk))
        seq_num += 1

    # Process remaining chunks
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg)
        del msg

        chunk, extra = await make_table_chunks_available_or_wait(
            context,
            chunk,
            reserve_extra=chunk.data_alloc_size(),
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            distinct_chunk = await asyncio.to_thread(
                _apply_distinct, chunk, ir, ir_context
            )
            del chunk

        if tracer is not None:
            tracer.add_chunk(table=distinct_chunk.table_view())
        await ch_out.send(context, Message(seq_num, distinct_chunk))
        seq_num += 1

    await ch_out.drain(context)


# ============================================================================
# Dynamic Distinct Node
# ============================================================================


@define_py_node()
async def distinct_node(
    context: Context,
    ir: Distinct,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    sample_chunk_count: int,
    target_partition_size: int,
    n_ary: int,
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
    n_ary
        Default fan-in for tree reduction (may be adapted based on chunk sizes).
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
        partitioned_inter_rank, partitioned_local = is_partitioned_on_keys(
            metadata_in, key_indices
        )

        nranks = context.comm().nranks

        # Determine if we can skip global communication
        can_skip_global_comm = (
            nranks == 1 or metadata_in.duplicated or partitioned_inter_rank
        )

        # If both inter-rank and local are partitioned, each chunk has
        # disjoint keys - we can do simple chunkwise distinct
        fully_partitioned = partitioned_inter_rank and partitioned_local

        # Sample chunks and apply local distinct
        initial_chunks: list[TableChunk] = []
        total_distinct_size = 0

        for _ in range(sample_chunk_count):
            msg = await ch_in.recv(context)
            if msg is None:
                break
            chunk = TableChunk.from_message(msg)
            del msg

            chunk, extra = await make_table_chunks_available_or_wait(
                context,
                chunk,
                reserve_extra=chunk.data_alloc_size(),
                net_memory_delta=0,
            )
            with opaque_memory_usage(extra):
                distinct_chunk = await asyncio.to_thread(
                    _apply_distinct, chunk, ir, ir_context
                )
                total_distinct_size += distinct_chunk.data_alloc_size(MemoryType.DEVICE)
                initial_chunks.append(distinct_chunk)
                del chunk

        # Estimate total size: avg_sample_size * local_count, summed across ranks
        local_count = metadata_in.local_count
        if initial_chunks and total_distinct_size > 0:
            avg_sample_size = total_distinct_size / len(initial_chunks)
            local_estimate = int(avg_sample_size * local_count)
            # Adaptive n-ary: how many chunks can fit in target_partition_size?
            # Bounded between 2 (minimum progress) and 256 (reasonable upper limit)
            chunks_per_partition = max(1, target_partition_size // avg_sample_size)
            adaptive_n_ary = max(2, min(256, int(chunks_per_partition)))
        else:
            local_estimate = 0
            adaptive_n_ary = n_ary  # fallback to configured value

        if collective_ids and nranks > 1:
            estimated_total_size, global_chunk_count = await allgather_reduce(
                context, collective_ids.pop(), local_estimate, local_count
            )
        else:
            estimated_total_size = local_estimate
            global_chunk_count = local_count

        # =====================================================================
        # Strategy Selection
        # =====================================================================

        if fully_partitioned:
            # Fully partitioned on keys - each chunk has disjoint keys
            # Just apply distinct to each chunk independently
            if tracer is not None:
                tracer.decision = "chunkwise"
            await _chunkwise_distinct(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                initial_chunks,
                tracer=tracer,
            )
        elif can_skip_global_comm:
            # No global communication needed
            if local_estimate < target_partition_size:
                # Small output - use local tree reduction
                if tracer is not None:
                    tracer.decision = "tree_local"
                await _tree_distinct(
                    context,
                    ir,
                    ir_context,
                    ch_out,
                    ch_in,
                    metadata_in,
                    initial_chunks,
                    adaptive_n_ary,
                    tracer=tracer,
                )
            else:
                # Large output - use local shuffle (no inter-rank communication)
                if tracer is not None:
                    tracer.decision = "shuffle_local"
                ideal_count = max(1, local_estimate // target_partition_size)
                output_count = max(1, min(ideal_count, local_count))
                await _shuffle_distinct(
                    context,
                    ir,
                    ir_context,
                    ch_out,
                    ch_in,
                    metadata_in,
                    initial_chunks,
                    output_count,
                    collective_ids.pop(),
                    key_indices,
                    tracer=tracer,
                )
        elif estimated_total_size < target_partition_size:
            # Small output - use tree reduction with allgather to merge across ranks
            if tracer is not None:
                tracer.decision = "tree_allgather"
            await _tree_distinct(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                initial_chunks,
                adaptive_n_ary,
                collective_id=collective_ids.pop(),
                tracer=tracer,
            )
        else:
            # Large output - use shuffle
            if tracer is not None:
                tracer.decision = "shuffle"
            ideal_count = max(1, estimated_total_size // target_partition_size)
            # Cap at global chunk count, but use the rank count if it's larger
            output_count = max(nranks, min(ideal_count, global_chunk_count))
            await _shuffle_distinct(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                initial_chunks,
                output_count,
                collective_ids.pop(),
                key_indices,
                shuffle_context=context,
                tracer=tracer,
            )


# ============================================================================
# Network Generation
# ============================================================================


@generate_ir_sub_network.register(Distinct)
def _(
    ir: Distinct, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]
    executor = config_options.executor

    if not isinstance(executor, StreamingExecutor) or executor.dynamic_planning is None:
        # Fall back to the default IR handler (bypass Distinct dispatch)
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    # For type narrowing after the early return
    dynamic_planning = executor.dynamic_planning

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Get collective IDs for this Distinct (may be empty if not reserved)
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))

    # Create the dynamic unique node
    nodes[ir] = [
        distinct_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            dynamic_planning.sample_chunk_count,
            executor.target_partition_size,
            executor.groupby_n_ary,  # Reuse groupby n_ary for now
            collective_ids,
        )
    ]

    return nodes, channels
