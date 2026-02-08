# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic Distinct (Unique) node for rapidsmpf runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Distinct
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
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

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer


def _apply_distinct(
    chunk: TableChunk,
    ir: Distinct,
    ir_context: Any,
) -> TableChunk:
    """Apply Distinct evaluation to a chunk."""
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
    tracer: ActorTracer | None = None,
) -> None:
    """
    Shuffle-based distinct.

    Shuffles data by distinct keys, then applies local distinct.
    """
    from cudf_polars.experimental.rapidsmpf.collectives.shuffle import (
        ShuffleManager,
    )

    nranks = context.comm().nranks

    # Calculate output partitioning
    modulus = nranks * output_count

    # Send output metadata
    from rapidsmpf.streaming.cudf.channel_metadata import HashScheme, Partitioning

    metadata_out = ChannelMetadata(
        local_count=output_count,
        partitioning=Partitioning(
            HashScheme(column_indices=key_indices, modulus=modulus),
            local="inherit",
        ),
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    # Create shuffle manager
    shuffle = ShuffleManager(context, output_count, key_indices, collective_id)

    # Insert initial chunks
    for chunk in initial_chunks:
        shuffle.insert_chunk(
            chunk.make_available_and_spill(context.br(), allow_overbooking=True)
        )

    # Insert remaining chunks from channel
    while (msg := await ch_in.recv(context)) is not None:
        shuffle.insert_chunk(
            TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
        )
        del msg

    await shuffle.insert_finished()

    # Extract shuffled partitions and apply local distinct
    input_schema = ir.children[0].schema
    stream = ir_context.get_cuda_stream()
    for seq_num, partition_id in enumerate(
        range(context.comm().rank, output_count, nranks)
    ):
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


# ============================================================================
# Dynamic Unique Node
# ============================================================================


@define_py_node()
async def unique_node(
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

    Strategy selection based on sampled data:
    - Skip global comm: Data already partitioned on keys or duplicated
    - Tree reduction: Small estimated output (< target_partition_size)
    - Shuffle: Large estimated output requiring redistribution
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
        already_partitioned = is_partitioned_on_keys(metadata_in, key_indices)

        nranks = context.comm().nranks

        # Determine if we can skip global communication
        can_skip_global_comm = (
            nranks == 1 or metadata_in.duplicated or already_partitioned
        )

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

        if already_partitioned or can_skip_global_comm:
            # No global communication needed - use tree reduction (no allgather)
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
                collective_ids.pop() if collective_ids else None,
                tracer,
            )
        elif not collective_ids:
            # No shuffle ID available - fall back to tree (no allgather)
            if tracer is not None:
                tracer.decision = "tree_fallback"
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
                tracer,
            )


# ============================================================================
# Network Generation
# ============================================================================


@generate_ir_sub_network.register(Distinct)
def _(
    ir: Distinct, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    from cudf_polars.utils.config import StreamingExecutor

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
        unique_node(
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
