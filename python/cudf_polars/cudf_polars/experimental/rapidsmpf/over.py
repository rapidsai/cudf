# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Window over() actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.expr import Col, GroupedWindow
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.ir import HStack
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.experimental.over import Over, _build_over_groupby_irs
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    NormalizedPartitioning,
    _evaluate_chunk_sync,
    allgather_reduce,
    chunk_to_frame,
    chunkwise_evaluate,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    maybe_remap_partitioning,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext, Select
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


def _broadcast_gw_sync(
    gw: GroupedWindow,
    chunk_df: DataFrame,
    global_agg_df: DataFrame,
    key_names: tuple[str, ...],
) -> Any:
    """Broadcast the global aggregate for one GroupedWindow back to row positions."""
    by_exprs = gw.children[: gw.by_count]
    by_cols = broadcast(
        *(b.evaluate(chunk_df) for b in by_exprs),
        target_length=chunk_df.num_rows,
        stream=chunk_df.stream,
    )
    by_tbl = plc.Table([c.obj for c in by_cols])
    group_keys_tbl = plc.Table(
        [global_agg_df.column_map[name].obj for name in key_names]
    )

    scalar_named, _ = gw._split_named_expr()
    _, out_names, out_dtypes = gw._build_groupby_requests(
        scalar_named, chunk_df, by_cols=by_cols
    )
    value_tbls = [
        plc.Table([global_agg_df.column_map[ne.name].obj]) for ne in scalar_named
    ]

    broadcasted_cols = gw._broadcast_agg_results(
        by_tbl, group_keys_tbl, value_tbls, out_names, out_dtypes, chunk_df.stream
    )
    temp_df = DataFrame(broadcasted_cols, stream=chunk_df.stream)
    return gw.post.value.evaluate(temp_df, context=ExecutionContext.FRAME)


def _evaluate_ir_broadcast_sync(
    chunk: TableChunk,
    ir: Select | HStack | Over,
    ir_context: IRExecutionContext,
    global_agg_df: DataFrame,
    key_names: tuple[str, ...],
    gw_nodes: tuple[GroupedWindow, ...],
) -> DataFrame:
    """Evaluate Select/HStack/Over using a pre-computed global aggregate for each GroupedWindow."""
    child_schema = ir.children[0].schema
    stream = ir_context.get_cuda_stream()
    chunk_df = DataFrame.from_table(
        chunk.table_view(),
        list(child_schema.keys()),
        list(child_schema.values()),
        stream,
    )

    gw_results: dict[int, Any] = {
        id(gw): _broadcast_gw_sync(gw, chunk_df, global_agg_df, key_names)
        for gw in gw_nodes
    }

    is_hstack = isinstance(ir, HStack)
    exprs = ir.columns if isinstance(ir, HStack) else ir.exprs
    result_cols = []
    for ne in exprs:
        if isinstance(ne.value, GroupedWindow) and id(ne.value) in gw_results:
            # gw.post.value.evaluate uses the post name, not ne.name
            col = gw_results[id(ne.value)].rename(ne.name)
        else:
            col = ne.evaluate(chunk_df, context=ExecutionContext.FRAME)
        result_cols.append(col)

    if is_hstack:
        return chunk_df.with_columns(result_cols, stream=stream)
    return DataFrame(result_cols, stream=stream)


async def _evaluate_broadcast_chunk(
    context: Context,
    chunk: TableChunk,
    ir: Select | HStack | Over,
    ir_context: IRExecutionContext,
    global_agg_df: DataFrame,
    key_names: tuple[str, ...],
    gw_nodes: tuple[GroupedWindow, ...],
) -> TableChunk:
    """Make chunk available then evaluate it against the pre-computed global aggregate."""
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=chunk.data_alloc_size(),
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        result_df = await asyncio.to_thread(
            _evaluate_ir_broadcast_sync,
            chunk,
            ir,
            ir_context,
            global_agg_df,
            key_names,
            gw_nodes,
        )
    return TableChunk.from_pylibcudf_table(
        result_df.table, result_df.stream, exclusive_view=True, br=context.br()
    )


_INT32_DTYPE = DataType(pl.Int32())
_INT64_DTYPE = DataType(pl.Int64())


@dataclass(frozen=True)
class OriginStamps:
    """
    Per-row metadata appended before shuffling so output can be reassembled.

    The forward shuffle distributes rows by group, breaking input order. After
    window evaluation we route rows back to their origin rank using ``rank``,
    then sort by ``(chunk_index, position)`` to recover the input chunking.

    ``chunk_index`` is a rank-local 0-based index into the stream of input
    chunks — *not* the upstream message ``sequence_number``, which is not
    guaranteed unique (e.g., when the input is the output of a shuffle whose
    partition IDs collide across chunks).
    """

    chunk_index: str
    position: str
    rank: str

    @property
    def names(self) -> tuple[str, str, str]:
        """Stamp column names, in the order they are appended to the table."""
        return (self.chunk_index, self.position, self.rank)

    @property
    def dtypes(self) -> tuple[DataType, DataType, DataType]:
        """Stamp column dtypes, parallel to :attr:`names`."""
        return (_INT32_DTYPE, _INT64_DTYPE, _INT32_DTYPE)


def _origin_stamps_for(input_ir: IR, ir: Over) -> OriginStamps:
    """Pick three stamp column names that do not collide with the schema."""
    names = unique_names((*input_ir.schema.keys(), *ir.schema.keys()))
    return OriginStamps(next(names), next(names), next(names))


def _append_origin_stamps(
    chunk: TableChunk,
    chunk_index: int,
    origin_rank: int,
    stream: Stream,
    br: Any,
) -> TableChunk:
    """Append (chunk_index, position, rank) stamp columns to *chunk*."""
    table = chunk.table_view()
    n_rows = table.num_rows()
    int32 = plc.types.DataType(plc.TypeId.INT32)
    int64 = plc.types.DataType(plc.TypeId.INT64)
    zero_step = plc.Scalar.from_py(0, int32, stream=stream)
    chunk_index_col = plc.filling.sequence(
        n_rows,
        plc.Scalar.from_py(chunk_index, int32, stream=stream),
        zero_step,
        stream=stream,
    )
    position_col = plc.filling.sequence(
        n_rows,
        plc.Scalar.from_py(0, int64, stream=stream),
        plc.Scalar.from_py(1, int64, stream=stream),
        stream=stream,
    )
    rank_col = plc.filling.sequence(
        n_rows,
        plc.Scalar.from_py(origin_rank, int32, stream=stream),
        zero_step,
        stream=stream,
    )
    return TableChunk.from_pylibcudf_table(
        plc.Table([*table.columns(), chunk_index_col, position_col, rank_col]),
        stream,
        exclusive_view=True,
        br=br,
    )


def _evaluate_window_with_stamps(
    chunk: TableChunk,
    ir: Over,
    ir_context: IRExecutionContext,
    stamps: OriginStamps,
) -> DataFrame:
    """Evaluate *ir* on a stamped chunk; the stamps ride through to the result."""
    child_schema = ir.children[0].schema
    stream = ir_context.get_cuda_stream()

    augmented = DataFrame.from_table(
        chunk.table_view(),
        [*child_schema.keys(), *stamps.names],
        [*child_schema.values(), *stamps.dtypes],
        stream,
    )
    result = ir.do_evaluate(
        ir.key_indices, ir.is_scalar, ir.exprs, augmented, context=ir_context
    )

    # ``do_evaluate`` returns only ir.exprs; reattach stamps from the input.
    missing = [
        augmented.column_map[name]
        for name in stamps.names
        if name not in result.column_map
    ]
    if missing:
        return DataFrame([*result.columns, *missing], stream=stream)
    return result


def _partition_by_origin_rank(
    result: DataFrame,
    num_ranks: int,
    br: Any,
) -> tuple[TableChunk | None, list[int]]:
    """
    Rearrange rows so partition i contains rows whose origin rank is i.

    Returns a chunk with the rank stamp dropped plus the split indices that
    feed straight into ``ShuffleManager.Inserter.insert_split``.
    """
    if result.table.num_rows() == 0:
        return None, []

    stream = result.stream
    columns = result.table.columns()
    rank_column = columns[-1]
    payload = plc.Table(columns[:-1])

    rearranged, offsets = plc.partitioning.partition(
        payload, rank_column, num_ranks, stream=stream
    )
    return (
        TableChunk.from_pylibcudf_table(rearranged, stream, exclusive_view=True, br=br),
        list(offsets[1:-1]),
    )


def _split_by_chunk_index(
    chunk: TableChunk,
    n_chunks: int,
    output_indices: list[int],
    chunk_index_column: int,
    position_column: int,
    ir_context: IRExecutionContext,
    br: Any,
) -> dict[int, TableChunk]:
    """
    Sort by ``(chunk_index, position)`` and split at chunk-index boundaries.

    Returns a mapping from chunk index in ``[0, n_chunks)`` to its rows.
    Chunk indices with no rows are absent from the result.
    """
    table = chunk.table_view()
    if table.num_rows() == 0:
        return {}

    stream = ir_context.get_cuda_stream()
    columns = table.columns()
    sort_order = plc.sorting.stable_sorted_order(
        plc.Table([columns[chunk_index_column], columns[position_column]]),
        [plc.types.Order.ASCENDING, plc.types.Order.ASCENDING],
        [plc.types.NullOrder.AFTER, plc.types.NullOrder.AFTER],
        stream=stream,
    )
    sorted_table = plc.copying.gather(
        table, sort_order, plc.copying.OutOfBoundsPolicy.DONT_CHECK, stream=stream
    )

    needles = plc.Column.from_iterable_of_py(
        list(range(1, n_chunks)),
        plc.types.DataType(plc.TypeId.INT32),
        stream=stream,
    )
    boundary_col = plc.search.lower_bound(
        plc.Table([sorted_table.columns()[chunk_index_column]]),
        plc.Table([needles]),
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.AFTER],
        stream=stream,
    )
    boundaries = (
        DataFrame.from_table(
            plc.Table([boundary_col]), ["p"], [_INT32_DTYPE], stream=stream
        )
        .to_polars()["p"]
        .to_list()
    )

    output_table = plc.Table([sorted_table.columns()[i] for i in output_indices])
    pieces = plc.copying.split(output_table, boundaries, stream=stream)

    by_chunk_index: dict[int, TableChunk] = {}
    for chunk_index, piece in enumerate(pieces):
        if piece.num_rows() == 0:
            continue
        # ``split`` returns zero-copy views into ``output_table``; concatenate
        # to materialise an independent buffer.
        by_chunk_index[chunk_index] = TableChunk.from_pylibcudf_table(
            plc.concatenate.concatenate([piece], stream=stream),
            stream,
            exclusive_view=True,
            br=br,
        )
    return by_chunk_index


def _empty_output_chunk(
    output_dtypes: list[DataType],
    stream: Stream,
    br: Any,
) -> TableChunk:
    """Build an empty chunk with the given column dtypes."""
    return TableChunk.from_pylibcudf_table(
        plc.Table(
            [
                plc.column_factories.make_empty_column(dt.plc_type, stream=stream)
                for dt in output_dtypes
            ]
        ),
        stream,
        exclusive_view=True,
        br=br,
    )


async def _allgather_and_broadcast(
    context: Context,
    comm: Communicator,
    ir: Over,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    input_ir: IR,
    tracer: Any,
    collective_id: int,
) -> None:
    """
    Compute partial aggregates per chunk, AllGather globally, then broadcast to each chunk.

    Buffers all incoming chunks while computing per-chunk partial aggregates,
    tree-reduces via AllGather, then broadcasts the global aggregate back to
    each buffered chunk using the pre-computed GroupedWindow lookup.
    """
    gw_nodes = tuple(ne.value for ne in ir.exprs if isinstance(ne.value, GroupedWindow))
    key_names = tuple(
        c.name
        for c in gw_nodes[0].children[: gw_nodes[0].by_count]
        if isinstance(c, Col)
    )
    piecewise_ir, reduction_ir, agg_select_ir = _build_over_groupby_irs(
        gw_nodes, input_ir
    )

    # make_table_chunks_available_or_wait releases the original chunk,
    # so we make each chunk available once and reuse it for both buffering
    # and piecewise evaluation.
    buffered: list[tuple[int, TableChunk]] = []
    partial_aggs: list[TableChunk] = []

    while (msg := await ch_in.recv(context)) is not None:
        raw_chunk = TableChunk.from_message(msg, br=context.br())
        avail_chunk, extra = await make_table_chunks_available_or_wait(
            context,
            raw_chunk,
            reserve_extra=raw_chunk.data_alloc_size(),
            net_memory_delta=0,
        )
        buffered.append((msg.sequence_number, avail_chunk))
        with opaque_memory_usage(extra):
            partial = await asyncio.to_thread(
                _evaluate_chunk_sync,
                avail_chunk,
                piecewise_ir,
                ir_context,
                context.br(),
            )
        partial_aggs.append(partial)

    if partial_aggs:
        local_agg = await evaluate_batch(
            partial_aggs, context, reduction_ir, ir_context=ir_context
        )
    else:
        local_agg = empty_table_chunk(
            reduction_ir, context, ir_context.get_cuda_stream()
        )

    # AllGather the unreduced form so the final reduction operates over
    # all ranks' partial results; agg_select_ir is applied only once after.
    if comm.nranks > 1 and not metadata_in.duplicated:
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            inserter.insert(0, local_agg)
        stream = ir_context.get_cuda_stream()
        concat_chunk = TableChunk.from_pylibcudf_table(
            await allgather.extract_concatenated(stream),
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        global_agg = await evaluate_chunk(
            context, concat_chunk, reduction_ir, ir_context=ir_context
        )
    else:
        global_agg = local_agg

    if agg_select_ir is not None:
        global_agg = await evaluate_chunk(
            context, global_agg, agg_select_ir, ir_context=ir_context
        )

    final_agg_ir = agg_select_ir if agg_select_ir is not None else reduction_ir
    global_agg_df = chunk_to_frame(global_agg, final_agg_ir)

    metadata_out = ChannelMetadata(
        local_count=metadata_in.local_count,
        partitioning=maybe_remap_partitioning(ir, metadata_in.partitioning),
        duplicated=metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    for seq_num, chunk in buffered:
        result = await _evaluate_broadcast_chunk(
            context,
            chunk,
            ir,
            ir_context,
            global_agg_df,
            key_names,
            gw_nodes,
        )
        if tracer is not None:
            tracer.add_chunk(table=result.table_view())
        await ch_out.send(context, Message(seq_num, result))

    await ch_out.drain(context)


async def _collect_input_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
) -> tuple[list[tuple[int, TableChunk]], int]:
    """Drain *ch_in* into spillable chunks, returning them with the byte total."""
    chunks: list[tuple[int, TableChunk]] = []
    local_bytes = 0
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg, br=context.br()).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        local_bytes += chunk.data_alloc_size()
        chunks.append((msg.sequence_number, chunk))
    return chunks, local_bytes


async def _negotiate_forward_modulus(
    context: Context,
    comm: Communicator,
    collective_id: int,
    local_bytes: int,
    local_count: int,
    target_partition_size: int,
) -> int:
    """AllGather global size+count and pick a partition count for the forward shuffle."""
    total_bytes, total_count = await allgather_reduce(
        context, comm, collective_id, local_bytes, local_count
    )
    return min(
        max(comm.nranks, total_bytes // max(1, target_partition_size)),
        max(1, total_count),
    )


async def _distribute_by_group(
    context: Context,
    comm: Communicator,
    forward_shuffle: ShuffleManager,
    inputs: list[tuple[int, TableChunk]],
    key_indices: tuple[int, ...],
    ir_context: IRExecutionContext,
    skip_insert: bool,  # noqa: FBT001
) -> None:
    """Stamp each chunk with origin metadata and hash-shuffle by partition keys."""
    async with forward_shuffle.inserting() as inserter:
        for chunk_index, (_, chunk) in enumerate(inputs):
            if skip_insert:
                continue
            stamped = await asyncio.to_thread(
                _append_origin_stamps,
                chunk,
                chunk_index,
                comm.rank,
                ir_context.get_cuda_stream(),
                context.br(),
            )
            inserter.insert_hash(stamped, key_indices)


async def _evaluate_and_route_to_origin(
    context: Context,
    ir: Over,
    ir_context: IRExecutionContext,
    forward_shuffle: ShuffleManager,
    return_shuffle: ShuffleManager,
    num_ranks: int,
    stamps: OriginStamps,
) -> None:
    """Window-evaluate each local forward partition, then ship rows back to their origin."""
    async with return_shuffle.inserting() as inserter:
        for partition_id in forward_shuffle.local_partitions():
            stream = ir_context.get_cuda_stream()
            extracted = forward_shuffle.extract_chunk(partition_id, stream)
            if extracted.num_rows() == 0:
                continue
            partition = TableChunk.from_pylibcudf_table(
                extracted, stream, exclusive_view=True, br=context.br()
            )
            evaluated = await asyncio.to_thread(
                _evaluate_window_with_stamps, partition, ir, ir_context, stamps
            )
            routed, splits = await asyncio.to_thread(
                _partition_by_origin_rank, evaluated, num_ranks, context.br()
            )
            if routed is not None:
                inserter.insert_split(routed, splits)


async def _collect_returned_rows(
    context: Context,
    return_shuffle: ShuffleManager,
    ir_context: IRExecutionContext,
) -> TableChunk | None:
    """Concatenate the local partitions of the return shuffle into a single chunk."""
    partition_ids = return_shuffle.local_partitions()
    if not partition_ids:
        return None
    stream = ir_context.get_cuda_stream()
    chunks = [
        TableChunk.from_pylibcudf_table(
            return_shuffle.extract_chunk(pid, stream),
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        for pid in partition_ids
    ]
    if len(chunks) == 1:
        return chunks[0]
    return TableChunk.from_pylibcudf_table(
        plc.concatenate.concatenate([c.table_view() for c in chunks], stream=stream),
        stream,
        exclusive_view=True,
        br=context.br(),
    )


async def _reassemble_input_chunks(
    context: Context,
    ch_out: Channel[TableChunk],
    ir_context: IRExecutionContext,
    received: TableChunk | None,
    sequence_numbers: list[int],
    ir: Over,
    tracer: Any,
) -> None:
    """Split received rows by chunk index and emit one output chunk per input chunk."""
    n_exprs = len(ir.exprs)
    output_indices = list(range(n_exprs))
    chunk_index_column = n_exprs
    position_column = n_exprs + 1
    output_dtypes = [ne.value.dtype for ne in ir.exprs]
    n_chunks = len(sequence_numbers)

    by_chunk_index: dict[int, TableChunk] = {}
    if received is not None and received.table_view().num_rows() > 0:
        by_chunk_index = await asyncio.to_thread(
            _split_by_chunk_index,
            received,
            n_chunks,
            output_indices,
            chunk_index_column,
            position_column,
            ir_context,
            context.br(),
        )

    stream = ir_context.get_cuda_stream()
    for chunk_index, sequence_number in enumerate(sequence_numbers):
        chunk = by_chunk_index.get(chunk_index)
        if chunk is None:
            chunk = await asyncio.to_thread(
                _empty_output_chunk, output_dtypes, stream, context.br()
            )
        if tracer is not None:
            tracer.add_chunk(table=chunk.table_view())
        await ch_out.send(context, Message(sequence_number, chunk))


async def _shuffle_and_reassemble(
    context: Context,
    comm: Communicator,
    ir: Over,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    input_ir: IR,
    tracer: Any,
    size_collective_id: int,
    forward_shuffle_collective_id: int,
    return_shuffle_collective_id: int,
    target_partition_size: int,
) -> None:
    """
    Evaluate non-decomposable window expressions across ranks via a return-shuffle.

    Each row is stamped with (origin rank, input sequence number, position in
    chunk). A hash shuffle by the partition-by keys co-locates each group on
    one rank for window evaluation; a second shuffle routes rows back to their
    origin so the rank that received an input chunk can also emit its output.
    Sorting the returned rows by (sequence, position) recovers input order.
    """
    stamps = _origin_stamps_for(input_ir, ir)

    metadata_out = ChannelMetadata(
        local_count=metadata_in.local_count,
        partitioning=maybe_remap_partitioning(ir, metadata_in.partitioning),
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    skip_insert = metadata_in.duplicated and comm.rank != 0

    inputs, local_bytes = await _collect_input_chunks(context, ch_in)
    forward_modulus = await _negotiate_forward_modulus(
        context,
        comm,
        size_collective_id,
        local_bytes,
        len(inputs),
        target_partition_size,
    )

    forward_shuffle = ShuffleManager(
        context, comm, forward_modulus, forward_shuffle_collective_id
    )
    return_shuffle = ShuffleManager(
        context,
        comm,
        comm.nranks,
        return_shuffle_collective_id,
        partition_assignment=PartitionAssignment.CONTIGUOUS,
    )

    sequence_numbers = [seq for seq, _ in inputs]
    await _distribute_by_group(
        context,
        comm,
        forward_shuffle,
        inputs,
        ir.key_indices,
        ir_context,
        skip_insert,
    )
    await _evaluate_and_route_to_origin(
        context,
        ir,
        ir_context,
        forward_shuffle,
        return_shuffle,
        comm.nranks,
        stamps,
    )
    received = await _collect_returned_rows(context, return_shuffle, ir_context)
    await _reassemble_input_chunks(
        context, ch_out, ir_context, received, sequence_numbers, ir, tracer
    )

    await ch_out.drain(context)


@define_actor()
async def over_actor(
    context: Context,
    comm: Communicator,
    ir: Over,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    collective_ids: list[int],
    target_partition_size: int,
) -> None:
    """
    Streaming actor for window ``over()`` expressions.

    Selects one of three strategies at runtime based on partitioning metadata
    and whether all GroupedWindow nodes are scalar aggregations: chunkwise
    (already partitioned), scalar broadcast (tree-reduce + AllGather), or
    non-scalar shuffle (hash-shuffle with row-index tracking for boundary
    reconstruction).

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Over IR node.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    collective_ids
        Collective IDs reserved for this operation. Scalar Over nodes receive
        one ID (AllGather); non-scalar nodes receive three (size AllGather +
        forward Shuffle + reverse Shuffle).
    target_partition_size
        Target output partition size in bytes, used to compute the shuffle
        modulus for the non-scalar path.
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        input_ir = ir.children[0]
        metadata_in = await recv_metadata(ch_in, context)

        partitioning = NormalizedPartitioning.from_keys(
            metadata_in.partitioning,
            comm.nranks,
            indices=ir.key_indices,
            allow_subset=False,
        )
        if partitioning:
            metadata_out = ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=maybe_remap_partitioning(ir, metadata_in.partitioning),
                duplicated=metadata_in.duplicated,
            )
            await chunkwise_evaluate(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_out,
                tracer=tracer,
            )
            return

        if ir.is_scalar:
            await _allgather_and_broadcast(
                context,
                comm,
                ir,
                ir_context,
                ch_in,
                ch_out,
                metadata_in,
                input_ir,
                tracer,
                collective_ids[0],
            )
        else:
            await _shuffle_and_reassemble(
                context,
                comm,
                ir,
                ir_context,
                ch_in,
                ch_out,
                metadata_in,
                input_ir,
                tracer,
                size_collective_id=collective_ids[0],
                forward_shuffle_collective_id=collective_ids[1],
                return_shuffle_collective_id=collective_ids[2],
                target_partition_size=target_partition_size,
            )


@generate_ir_sub_network.register(Over)
def _(
    ir: Over, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    actors[ir] = [
        over_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_ids,
            config_options.executor.target_partition_size,
        )
    ]
    return actors, channels
