# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Window ``over()`` actor for the RapidsMPF streaming runtime.

Implements the ``group_to_rows`` ``WindowMapping`` only: each input row
receives the value computed for its group. Other mappings (``explode``,
``join``) are not supported.

The actor picks one of three strategies at runtime based on the incoming
channel metadata and the shape of the windowed expressions.

Chunkwise (already partitioned)
    If the channel is already hash-partitioned on the over-keys (or any
    prefix of them), every group is fully contained within one rank's
    chunks. The window expression is correct on each chunk in isolation
    and no cross-rank coordination is needed.

Scalar broadcast (decomposable aggregations)
    When every aggregation is decomposable, partial aggregates can be
    combined associatively across ranks. Each rank computes per-chunk
    partials, an AllGather collects them, a single reduction yields the
    global aggregate per group, and each input chunk has those results
    joined back onto its rows by the partition keys. Order is preserved
    naturally: input chunks are buffered in receive order and emitted in
    the same order after the global aggregate is known.

Forward + return shuffle (non-decomposable aggregations)
    For functions that need every row in a group visible at once, a hash
    shuffle on the partition keys co-locates each group on one rank for
    evaluation. After evaluation, a second shuffle routes each row back
    to the rank that originally received it (output channels are
    rank-local, so only the originating rank can emit), and the rows are
    reassembled in input order using stamps that travel with the data
    through both shuffles. The forward shuffle is sized by sampling a few
    input chunks so insertion can stream rather than buffering everything.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

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

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl.expr import GroupedWindow
from cudf_polars.dsl.expressions.base import ExecutionContext
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
    _sample_chunks,
    allgather_reduce,
    chunk_to_frame,
    chunkwise_evaluate,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    gather_in_task_group,
    maybe_remap_partitioning,
    process_children,
    recv_metadata,
    replay_buffered_channel,
    send_metadata,
    shutdown_on_error,
)

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.expr import Col
    from cudf_polars.dsl.ir import IR, GroupBy, IRExecutionContext, Select
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import TableSizeStats


@dataclass(frozen=True)
class _ScalarOverPlan:
    """Pre-computed IR rewrites for the scalar Over path."""

    gw_nodes: tuple[GroupedWindow, ...]
    key_names: tuple[str, ...]
    piecewise_ir: GroupBy
    reduction_ir: GroupBy
    agg_select_ir: Select


def _build_scalar_over_plan(ir: Over) -> _ScalarOverPlan:
    """Pre-compute the IR rewrites needed by the scalar Over path."""
    gw_nodes = tuple(ne.value for ne in ir.exprs if isinstance(ne.value, GroupedWindow))
    # Lowering rejects non-Col partition-by keys, so every by-child here is a Col.
    by_children = gw_nodes[0].children[: gw_nodes[0].by_count]
    key_names = tuple(cast("Col", c).name for c in by_children)
    piecewise_ir, reduction_ir, agg_select_ir = _build_over_groupby_irs(
        gw_nodes, ir.children[0]
    )
    return _ScalarOverPlan(
        gw_nodes=gw_nodes,
        key_names=key_names,
        piecewise_ir=piecewise_ir,
        reduction_ir=reduction_ir,
        agg_select_ir=agg_select_ir,
    )


def _broadcast_gw_sync(
    gw: GroupedWindow,
    chunk_df: DataFrame,
    global_agg_df: DataFrame,
    key_names: tuple[str, ...],
    stream: Stream,
) -> Any:
    """Broadcast the global aggregate for one GroupedWindow back to row positions."""
    by_exprs = gw.children[: gw.by_count]
    by_cols = broadcast(
        *(b.evaluate(chunk_df) for b in by_exprs),
        target_length=chunk_df.num_rows,
        stream=stream,
    )
    by_tbl = plc.Table([c.obj for c in by_cols])
    group_keys_tbl = global_agg_df.select(key_names).table

    out_names, out_dtypes = zip(
        *((ne.name, ne.value.dtype) for ne in gw.named_aggs), strict=True
    )
    value_tbls = [
        plc.Table([global_agg_df.column_map[ne.name].obj]) for ne in gw.named_aggs
    ]

    broadcasted_cols = gw._broadcast_agg_results(
        by_tbl, group_keys_tbl, value_tbls, out_names, out_dtypes, stream
    )
    temp_df = DataFrame(broadcasted_cols, stream=stream)
    return gw.post.value.evaluate(temp_df, context=ExecutionContext.FRAME)


def _evaluate_ir_broadcast_sync(
    chunk: TableChunk,
    ir: Over,
    global_agg_df: DataFrame,
    key_names: tuple[str, ...],
    gw_nodes: tuple[GroupedWindow, ...],
    ir_context: IRExecutionContext,
    br: BufferResource,
) -> TableChunk:
    """Evaluate the Over node using a pre-computed global aggregate per GroupedWindow."""
    chunk_df = chunk_to_frame(chunk, ir.children[0])
    # global_agg_df and chunk_df may live on different streams (the former from
    # the upstream allgather/reduction on ir_context's stream, the latter from
    # the input message). Join them so the broadcast kernels read global_agg_df
    # safely.
    with ir_context.stream_ordered_after(chunk_df, global_agg_df) as stream:
        chunk_df = DataFrame(chunk_df.columns, stream=stream)
        global_agg_df = DataFrame(global_agg_df.columns, stream=stream)

        gw_results = {
            gw: _broadcast_gw_sync(gw, chunk_df, global_agg_df, key_names, stream)
            for gw in gw_nodes
        }

        result_cols = []
        for ne in ir.exprs:
            if isinstance(ne.value, GroupedWindow):
                # gw.post.value.evaluate uses the post name, not ne.name
                col = gw_results[ne.value].rename(ne.name)
            else:
                col = ne.evaluate(chunk_df, context=ExecutionContext.FRAME)
            result_cols.append(col)

        return TableChunk.from_pylibcudf_table(
            plc.Table([c.obj for c in result_cols]),
            stream,
            exclusive_view=True,
            br=br,
        )


async def _evaluate_broadcast_chunk(
    context: Context,
    chunk: TableChunk,
    ir: Over,
    global_agg_df: DataFrame,
    key_names: tuple[str, ...],
    gw_nodes: tuple[GroupedWindow, ...],
    ir_context: IRExecutionContext,
) -> TableChunk:
    """Make chunk available then evaluate it against the pre-computed global aggregate."""
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=chunk.data_alloc_size(),
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        return await asyncio.to_thread(
            _evaluate_ir_broadcast_sync,
            chunk,
            ir,
            global_agg_df,
            key_names,
            gw_nodes,
            ir_context,
            context.br(),
        )


@dataclass(frozen=True)
class OriginStamps:
    """
    Stamp column names that ride both shuffles for output reassembly.

    Parameters
    ----------
    chunk_index
        Column name for a dense rank-local 0..N-1 counter identifying which
        input chunk a row came from.
    position
        Column name for the row's position within its input chunk.
    rank
        Column name for the originating rank.
    """

    chunk_index: str
    position: str
    rank: str

    dtype: ClassVar[DataType] = DataType(pl.Int32())

    @property
    def names(self) -> tuple[str, str, str]:
        """Stamp column names, in the order they are appended to the table."""
        return (self.chunk_index, self.position, self.rank)


def _origin_stamps_for(ir: Over) -> OriginStamps:
    """Pick three stamp column names that do not collide with the schema."""
    names = unique_names((*ir.children[0].schema.keys(), *ir.schema.keys()))
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
    chunk_index_col = plc.Column.from_scalar(
        plc.Scalar.from_py(chunk_index, int32, stream=stream), n_rows, stream=stream
    )
    rank_col = plc.Column.from_scalar(
        plc.Scalar.from_py(origin_rank, int32, stream=stream), n_rows, stream=stream
    )
    position_col = plc.filling.sequence(
        n_rows,
        plc.Scalar.from_py(0, int32, stream=stream),
        plc.Scalar.from_py(1, int32, stream=stream),
        stream=stream,
    )
    return TableChunk.from_pylibcudf_table(
        plc.Table([*table.columns(), chunk_index_col, position_col, rank_col]),
        stream,
        exclusive_view=False,
        br=br,
    )


def _evaluate_window_with_stamps(
    chunk: TableChunk,
    ir: Over,
    ir_context: IRExecutionContext,
    stamps: OriginStamps,
) -> DataFrame:
    """Evaluate *ir* on the un-stamped portion of *chunk*; reattach stamps after."""
    child_schema = ir.children[0].schema
    stream = ir_context.get_cuda_stream()
    columns = chunk.table_view().columns()
    n_child = len(child_schema)

    input_df = DataFrame.from_table(
        plc.Table(columns[:n_child]),
        list(child_schema.keys()),
        list(child_schema.values()),
        stream,
    )
    result = ir.do_evaluate(ir.exprs, input_df, context=ir_context)
    stamp_cols = [
        Column(col, dtype=stamps.dtype, name=name)
        for col, name in zip(columns[n_child:], stamps.names, strict=True)
    ]
    return result.with_columns(stamp_cols, stream=stream)


def _partition_by_origin_rank(
    result: DataFrame,
    num_ranks: int,
    br: Any,
) -> tuple[TableChunk | None, list[int]]:
    """
    Rearrange rows so partition i contains rows whose origin rank is i.

    Returns a chunk with the rank stamp dropped and the per-rank split
    indices for direct insertion into the return shuffle.
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
    Sort by ``(chunk_index, position)`` and split at chunk-index transitions.

    Returns a mapping from chunk index in ``[0, n_chunks)`` to its rows.
    Chunk indices with no rows are absent from the result.
    """
    table = chunk.table_view()
    if table.num_rows() == 0:
        return {}

    stream = ir_context.get_cuda_stream()
    columns = table.columns()
    sorted_table = plc.sorting.stable_sort_by_key(
        table,
        plc.Table([columns[chunk_index_column], columns[position_column]]),
        [plc.types.Order.ASCENDING, plc.types.Order.ASCENDING],
        [plc.types.NullOrder.AFTER, plc.types.NullOrder.AFTER],
        stream=stream,
    )

    needles = plc.Column.from_iterable_of_py(
        range(1, n_chunks),
        plc.types.DataType(plc.TypeId.INT32),
        stream=stream,
    )
    split_position_col = plc.search.lower_bound(
        plc.Table([sorted_table.columns()[chunk_index_column]]),
        plc.Table([needles]),
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.AFTER],
        stream=stream,
    )
    split_positions = (
        DataFrame.from_table(
            plc.Table([split_position_col]),
            ["p"],
            [DataType(pl.Int32())],
            stream=stream,
        )
        .to_polars()["p"]
        .to_list()
    )

    output_table = plc.Table([sorted_table.columns()[i] for i in output_indices])
    pieces = plc.copying.split(output_table, split_positions, stream=stream)

    by_chunk_index: dict[int, TableChunk] = {}
    for chunk_index, piece in enumerate(pieces):
        if piece.num_rows() == 0:
            continue
        # ``split`` returns zero-copy views into ``output_table``; copy to
        # materialise an independent buffer.
        by_chunk_index[chunk_index] = TableChunk.from_pylibcudf_table(
            piece.copy(stream=stream),
            stream,
            exclusive_view=True,
            br=br,
        )
    return by_chunk_index


async def _allgather_and_broadcast(
    context: Context,
    comm: Communicator,
    ir: Over,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    tracer: Any,
    collective_id: int,
    plan: _ScalarOverPlan,
) -> None:
    """Compute partial aggregates per chunk, AllGather globally, then broadcast to each chunk."""
    piecewise_ir = plan.piecewise_ir
    reduction_ir = plan.reduction_ir
    agg_select_ir = plan.agg_select_ir

    # Making the chunk available consumes the spilled handle, so we keep
    # the available chunk and reuse it for both buffering and piecewise eval.
    buffered: list[tuple[int, TableChunk]] = []
    partial_aggs: list[TableChunk] = []

    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg, br=context.br())
        chunk, extra = await make_table_chunks_available_or_wait(
            context,
            chunk,
            reserve_extra=chunk.data_alloc_size(),
            net_memory_delta=0,
        )
        buffered.append((msg.sequence_number, chunk))
        with opaque_memory_usage(extra):
            partial = await asyncio.to_thread(
                _evaluate_chunk_sync,
                chunk,
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
    # all ranks' partials; the post-aggregation step runs once after.
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

    global_agg = await evaluate_chunk(
        context, global_agg, agg_select_ir, ir_context=ir_context
    )
    global_agg_df = chunk_to_frame(global_agg, agg_select_ir)

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
            global_agg_df,
            plan.key_names,
            plan.gw_nodes,
            ir_context,
        )
        if tracer is not None:
            tracer.add_chunk(table=result.table_view())
        await ch_out.send(context, Message(seq_num, result))

    await ch_out.drain(context)


async def _choose_modulus(
    context: Context,
    comm: Communicator,
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    collective_id: int,
    target_partition_size: int,
    sample_chunk_count: int,
) -> tuple[TableSizeStats, int]:
    """
    Sample input, AllGather size estimates, and derive the forward-shuffle modulus.

    Returns the sample (whose chunks must be replayed back to the consumer)
    and the chosen number of forward-shuffle partitions.
    """
    sample = await _sample_chunks(
        context,
        ch_in,
        sample_chunk_count,
        target_partition_size,
        metadata_in.local_count,
    )
    if comm.nranks > 1 and not metadata_in.duplicated:
        total_bytes, total_count = await allgather_reduce(
            context, comm, collective_id, sample.total_size, sample.total_chunks
        )
    else:
        total_bytes, total_count = sample.total_size, sample.total_chunks
    modulus = min(
        max(comm.nranks, total_bytes // max(1, target_partition_size)),
        max(1, total_count),
    )
    return sample, modulus


async def _distribute_by_group(
    context: Context,
    comm: Communicator,
    forward_shuffle: ShuffleManager,
    ch_in: Channel[TableChunk],
    key_indices: tuple[int, ...],
    ir_context: IRExecutionContext,
    skip_insert: bool,  # noqa: FBT001
) -> list[int]:
    """Stream chunks from *ch_in* into the forward shuffle with origin stamps."""
    # The replay channel re-sends metadata; we already have it, so discard.
    await recv_metadata(ch_in, context)

    sequence_numbers: list[int] = []
    chunk_index = 0
    async with forward_shuffle.inserting() as inserter:
        while (msg := await ch_in.recv(context)) is not None:
            chunk = TableChunk.from_message(
                msg, br=context.br()
            ).make_available_and_spill(context.br(), allow_overbooking=True)
            sequence_numbers.append(msg.sequence_number)
            if not skip_insert:
                stamped = await asyncio.to_thread(
                    _append_origin_stamps,
                    chunk,
                    chunk_index,
                    comm.rank,
                    ir_context.get_cuda_stream(),
                    context.br(),
                )
                inserter.insert_hash(stamped, key_indices)
            chunk_index += 1
    return sequence_numbers


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
        chunk = by_chunk_index.get(chunk_index) or empty_table_chunk(
            ir, context, stream
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
    tracer: Any,
    size_collective_id: int,
    forward_shuffle_collective_id: int,
    return_shuffle_collective_id: int,
    target_partition_size: int,
    sample_chunk_count: int,
) -> None:
    """Hash-shuffle by partition keys, evaluate, then route rows back to their origin rank."""
    stamps = _origin_stamps_for(ir)

    metadata_out = ChannelMetadata(
        local_count=metadata_in.local_count,
        partitioning=maybe_remap_partitioning(ir, metadata_in.partitioning),
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    skip_insert = metadata_in.duplicated and comm.rank != 0

    sample, forward_modulus = await _choose_modulus(
        context,
        comm,
        ch_in,
        metadata_in,
        size_collective_id,
        target_partition_size,
        sample_chunk_count,
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

    ch_replay = context.create_channel()
    sequence_numbers, _ = await gather_in_task_group(
        _distribute_by_group(
            context,
            comm,
            forward_shuffle,
            ch_replay,
            ir.key_indices,
            ir_context,
            skip_insert,
        ),
        replay_buffered_channel(
            context, ch_replay, ch_in, sample.chunks, metadata_in, trace_ir=ir
        ),
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
    sample_chunk_count: int,
    scalar_plan: _ScalarOverPlan | None,
) -> None:
    """
    Streaming actor for window ``over()`` expressions.

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
        one ID (AllGather); non-scalar nodes receive two (one shared by the
        size AllGather and forward Shuffle, plus a separate one for the
        return Shuffle which overlaps with the forward extract).
    target_partition_size
        Target output partition size in bytes, used to compute the shuffle
        modulus for the non-scalar path.
    sample_chunk_count
        Maximum number of input chunks to sample when estimating the shuffle
        modulus on the non-scalar path.
    scalar_plan
        Pre-computed IR rewrites for the scalar Over path, built at planning
        time. ``None`` for non-scalar Over nodes.
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        metadata_in = await recv_metadata(ch_in, context)

        partitioning = NormalizedPartitioning.from_keys(
            metadata_in.partitioning,
            comm.nranks,
            keys=ir.key_indices,
            allow_subset=True,
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
            assert scalar_plan is not None
            await _allgather_and_broadcast(
                context,
                comm,
                ir,
                ir_context,
                ch_in,
                ch_out,
                metadata_in,
                tracer,
                collective_ids[0],
                scalar_plan,
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
                tracer,
                # collective_ids[0] is reused for the size AllGather and the
                # forward shuffle (sequential, no overlap); collective_ids[1]
                # is the return shuffle, which overlaps with forward extract.
                size_collective_id=collective_ids[0],
                forward_shuffle_collective_id=collective_ids[0],
                return_shuffle_collective_id=collective_ids[1],
                target_partition_size=target_partition_size,
                sample_chunk_count=sample_chunk_count,
            )


@generate_ir_sub_network.register(Over)
def _(
    ir: Over, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    executor = rec.state["config_options"].executor
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    if not ir.is_scalar and executor.dynamic_planning is None:
        raise ValueError(
            "Non-scalar over() requires dynamic planning to size the forward "
            "shuffle. Enable it via StreamingExecutor(dynamic_planning=...) "
            "or the --dynamic-planning CLI flag."
        )
    sample_chunk_count = (
        executor.dynamic_planning.sample_chunk_count
        if executor.dynamic_planning is not None
        else 0
    )
    scalar_plan = _build_scalar_over_plan(ir) if ir.is_scalar else None
    actors[ir] = [
        over_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_ids,
            executor.target_partition_size,
            sample_chunk_count,
            scalar_plan,
        )
    ]
    return actors, channels
