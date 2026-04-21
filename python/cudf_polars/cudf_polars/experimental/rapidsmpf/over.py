# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Window over() actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rapidsmpf.memory.memory_reservation import opaque_memory_usage
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
from cudf_polars.dsl.expr import GroupedWindow
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.ir import IR, HStack, Select
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.experimental.over import Over
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    NormalizedPartitioning,
    _evaluate_chunk_sync,
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
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
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
    ir: Select | HStack,
    ir_context: IRExecutionContext,
    global_agg_df: DataFrame,
    key_names: tuple[str, ...],
    gw_nodes: tuple[GroupedWindow, ...],
) -> DataFrame:
    """Evaluate Select/HStack using a pre-computed global aggregate for each GroupedWindow."""
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
    exprs = ir.exprs if isinstance(ir, Select) else ir.columns
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
    ir: Select | HStack,
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


_INT64_DTYPE = DataType(pl.Int64())


def _add_row_idx_sync(
    chunk: TableChunk,
    row_start: int,
    stream: Stream,
    br: Any,
) -> TableChunk:
    """Append a row-index column to *chunk* (chunk must be available)."""
    tbl = chunk.table_view()
    n_rows = tbl.num_rows()
    row_idx = plc.filling.sequence(
        n_rows,
        plc.Scalar.from_py(
            row_start, plc.types.DataType(plc.TypeId.INT64), stream=stream
        ),
        plc.Scalar.from_py(1, plc.types.DataType(plc.TypeId.INT64), stream=stream),
        stream=stream,
    )
    return TableChunk.from_pylibcudf_table(
        plc.Table([*tbl.columns(), row_idx]),
        stream,
        exclusive_view=True,
        br=br,
    )


def _evaluate_with_row_idx_sync(
    chunk: TableChunk,
    ir: Select | HStack,
    ir_context: IRExecutionContext,
    row_idx_col: str,
) -> DataFrame:
    """Evaluate ir on *chunk* augmented with *row_idx_col*; return result with row_idx attached."""
    child_schema = ir.children[0].schema
    stream = ir_context.get_cuda_stream()

    augmented_keys = [*child_schema.keys(), row_idx_col]
    augmented_vals = [*child_schema.values(), _INT64_DTYPE]
    augmented_df = DataFrame.from_table(
        chunk.table_view(), augmented_keys, augmented_vals, stream
    )

    result_df = ir.do_evaluate(*ir._non_child_args, augmented_df, context=ir_context)

    # HStack passes row_idx_col through; Select does not — attach it from augmented_df.
    if row_idx_col not in result_df.column_map:
        idx_col = augmented_df.column_map[row_idx_col]
        return DataFrame([*result_df.columns, idx_col], stream=stream)
    return result_df


def _sort_df_by_row_idx_sync(
    result_df: DataFrame,
    col_names: list[str],
    col_types: list[DataType],
    row_idx_col: str,
) -> DataFrame:
    """Sort *result_df* by its row_idx column and return a sorted DataFrame."""
    idx_col_idx = col_names.index(row_idx_col)
    stream = result_df.stream
    sort_order = plc.sorting.stable_sorted_order(
        plc.Table([result_df.table.columns()[idx_col_idx]]),
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.AFTER],
        stream=stream,
    )
    sorted_tbl = plc.copying.gather(
        result_df.table,
        sort_order,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )
    return DataFrame.from_table(sorted_tbl, col_names, col_types, stream)


def _slice_and_accumulate_sync(
    sorted_df: DataFrame,
    split_values: list[int],
    n_boundaries: int,
    col_names: list[str],
    col_types: list[DataType],
    idx_col_idx: int,
    br: Any,
    accumulated: list[list[TableChunk]],
) -> None:
    """
    Slice *sorted_df* into per-boundary sub-chunks and append to *accumulated*.

    Uses ``lower_bound`` to find all split positions in one GPU call, then
    ``plc.copying.split`` for zero-copy views.  Each non-empty sub-view is
    materialised into an independent ``TableChunk`` (registered with *br* so
    the SpillManager can move it to host under memory pressure).

    Parameters
    ----------
    sorted_df
        One partition sorted by row_idx.
    split_values
        Start row_idx of each boundary except the first; len = n_boundaries - 1.
    n_boundaries
        Total number of output boundaries.
    col_names, col_types
        Column names/types for the result DataFrame.
    idx_col_idx
        Column index of the row_idx column.
    br
        RapidsMPF BufferResource for spill registration.
    accumulated
        Output list-of-lists; accumulated[k] receives sub-chunks for boundary k.
    """
    n_rows = sorted_df.table.num_rows()
    if n_rows == 0:
        return

    stream = sorted_df.stream

    if split_values:
        needles_col = plc.Column.from_iterable_of_py(
            split_values,
            plc.types.DataType(plc.TypeId.INT64),
            stream=stream,
        )
        split_pos_col = plc.search.lower_bound(
            plc.Table([sorted_df.table.columns()[idx_col_idx]]),
            plc.Table([needles_col]),
            [plc.types.Order.ASCENDING],
            [plc.types.NullOrder.AFTER],
            stream=stream,
        )
        split_pts = (
            DataFrame.from_table(
                plc.Table([split_pos_col]),
                ["p"],
                [DataType(pl.Int32())],
                stream=stream,
            )
            .to_polars()["p"]
            .to_list()
        )
    else:
        split_pts = []

    sub_views = plc.copying.split(sorted_df.table, split_pts, stream=stream)

    for k, sv in enumerate(sub_views):
        if sv.num_rows() > 0:
            # Materialise: plc.copying.split returns zero-copy views into sorted_df.
            # We need an independent buffer so sorted_df can be freed and the
            # sub-chunk remains valid (and spill-eligible).
            chunk = TableChunk.from_pylibcudf_table(
                plc.concatenate.concatenate([sv], stream=stream),
                stream,
                exclusive_view=True,
                br=br,
            )
            accumulated[k].append(chunk)


def _concat_sort_boundary_sync(
    sub_chunks: list[TableChunk],
    col_names: list[str],
    col_types: list[DataType],
    idx_col_idx: int,
    keep_col_indices: list[int],
    ir_context: IRExecutionContext,
    br: Any,
) -> TableChunk:
    """
    Concatenate and sort *sub_chunks* for one boundary, stripping the row_idx column.

    Parameters
    ----------
    sub_chunks
        Per-partition sub-chunks for this boundary, already made available on device.
    col_names, col_types
        Column names/types (including row_idx).
    idx_col_idx
        Column index of the row_idx column.
    keep_col_indices
        Column indices to keep in the output (excludes row_idx).
    ir_context
        Execution context (provides CUDA stream and memory resource).
    br
        RapidsMPF BufferResource for output chunk registration.
    """
    stream = ir_context.get_cuda_stream()
    if not sub_chunks:
        return TableChunk.from_pylibcudf_table(
            plc.Table(
                [
                    plc.column_factories.make_empty_column(
                        col_types[i].plc_type, stream=stream
                    )
                    for i in keep_col_indices
                ]
            ),
            stream,
            exclusive_view=True,
            br=br,
        )

    sub_dfs = [
        DataFrame.from_table(chunk.table_view(), col_names, col_types, chunk.stream)
        for chunk in sub_chunks
    ]
    merged_df = (
        _concat(*sub_dfs, context=ir_context) if len(sub_dfs) > 1 else sub_dfs[0]
    )
    merged = merged_df.table
    stream = merged_df.stream

    sort_order = plc.sorting.stable_sorted_order(
        plc.Table([merged.columns()[idx_col_idx]]),
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.AFTER],
        stream=stream,
    )
    sorted_tbl = plc.copying.gather(
        plc.Table([merged.columns()[i] for i in keep_col_indices]),
        sort_order,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )
    return TableChunk.from_pylibcudf_table(
        sorted_tbl, stream, exclusive_view=True, br=br
    )


@define_actor()
async def over_actor(
    context: Context,
    comm: Communicator,
    ir: Over,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    collective_id: int,
) -> None:
    """
    Streaming actor for window ``over()`` expressions.

    Selects one of three strategies at runtime based on partitioning metadata
    and whether all GroupedWindow nodes are scalar aggregations: chunkwise
    (already partitioned), scalar broadcast (tree-reduce + AllGather), or
    non-scalar shuffle (hash-shuffle with optional row-index tracking for HStack).

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Over IR node carrying the wrapped Select/HStack and pre-computed metadata.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    collective_id
        Collective ID reserved for this operation (AllGather or Shuffle).
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        wrapped_ir: Select | HStack = ir.children[0]  # type: ignore[assignment]
        metadata_in = await recv_metadata(ch_in, context)

        key_indices = ir.key_indices
        assert len(key_indices) > 0

        partitioning = NormalizedPartitioning.from_indices(
            metadata_in.partitioning,
            comm.nranks,
            indices=key_indices,
            allow_subset=False,
        )
        if partitioning:
            metadata_out = ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=maybe_remap_partitioning(
                    wrapped_ir, metadata_in.partitioning
                ),
                duplicated=metadata_in.duplicated,
            )
            await chunkwise_evaluate(
                context,
                wrapped_ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_out,
                tracer=tracer,
            )
            return

        if ir.is_scalar:
            gw_nodes = ir.gw_nodes
            key_names = ir.key_names
            piecewise_ir = ir.piecewise_ir
            reduction_ir = ir.reduction_ir
            agg_select_ir = ir.agg_select_ir
            assert gw_nodes is not None
            assert key_names is not None
            assert piecewise_ir is not None
            assert reduction_ir is not None

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
                partitioning=maybe_remap_partitioning(
                    wrapped_ir, metadata_in.partitioning
                ),
                duplicated=metadata_in.duplicated,
            )
            await send_metadata(ch_out, context, metadata_out)

            for seq_num, chunk in buffered:
                result = await _evaluate_broadcast_chunk(
                    context,
                    chunk,
                    wrapped_ir,
                    ir_context,
                    global_agg_df,
                    key_names,
                    gw_nodes,
                )
                if tracer is not None:
                    tracer.add_chunk(table=result.table_view())
                await ch_out.send(context, Message(seq_num, result))

            await ch_out.drain(context)
            return

        row_idx_col = ir.row_idx_col
        assert row_idx_col is not None

        modulus = max(comm.nranks, metadata_in.local_count)
        metadata_out = ChannelMetadata(
            local_count=metadata_in.local_count,
            partitioning=maybe_remap_partitioning(wrapped_ir, metadata_in.partitioning),
            duplicated=metadata_in.duplicated,
        )
        await send_metadata(ch_out, context, metadata_out)

        shuffle = ShuffleManager(context, comm, modulus, collective_id)
        row_counter = 0
        boundaries: list[tuple[int, int, int]] = []

        # Phase 1: stamp each row with its absolute position and insert into the shuffle
        async with shuffle.inserting() as inserter:
            while (msg := await ch_in.recv(context)) is not None:
                chunk = TableChunk.from_message(
                    msg, br=context.br()
                ).make_available_and_spill(context.br(), allow_overbooking=True)
                stream = ir_context.get_cuda_stream()
                n_rows = chunk.table_view().num_rows()
                boundaries.append(
                    (msg.sequence_number, row_counter, row_counter + n_rows)
                )
                chunk = await asyncio.to_thread(
                    _add_row_idx_sync, chunk, row_counter, stream, context.br()
                )
                row_counter += n_rows
                inserter.insert_hash(chunk, key_indices)

        if not boundaries:
            await ch_out.drain(context)
            return

        # Three-phase sort-and-split.
        #
        # Phase 1 (insert): stamp each row with its absolute position and insert
        #   into the shuffle (see the inserting() block above).
        # Phase 2 (per partition): extract, evaluate, sort by row_idx, then
        #   immediately slice into per-boundary sub-chunks and materialise each
        #   as an independent TableChunk registered with br.  The sorted
        #   partition is freed as soon as its sub-chunks are accumulated, so the
        #   SpillManager can evict earlier sub-chunks to host if memory is tight.
        # Phase 3 (per boundary): bring sub-chunks back to device if spilled
        #   (make_available_or_wait), concat + sort + output.
        #
        # This keeps total working memory proportional to one boundary's output
        # at a time (not all sorted partitions simultaneously).
        col_names: list[str] | None = None
        col_types: list[DataType] | None = None
        idx_col_idx: int = -1
        keep_col_indices: list[int] = []
        split_values: list[int] = [start for _, start, _ in boundaries[1:]]
        accumulated: list[list[TableChunk]] = [[] for _ in boundaries]

        # Phase 2
        for partition_id in shuffle.local_partitions():
            stream = ir_context.get_cuda_stream()
            partition_chunk = TableChunk.from_pylibcudf_table(
                shuffle.extract_chunk(partition_id, stream),
                stream,
                exclusive_view=True,
                br=context.br(),
            )
            result_df = await asyncio.to_thread(
                _evaluate_with_row_idx_sync,
                partition_chunk,
                wrapped_ir,
                ir_context,
                row_idx_col,
            )
            if col_names is None:
                col_names = [c.name for c in result_df.columns]
                col_types = [c.dtype for c in result_df.columns]
                idx_col_idx = col_names.index(row_idx_col)
                keep_col_indices = [
                    i for i, n in enumerate(col_names) if n != row_idx_col
                ]
            assert col_types is not None
            sorted_df = await asyncio.to_thread(
                _sort_df_by_row_idx_sync,
                result_df,
                col_names,
                col_types,
                row_idx_col,
            )
            await asyncio.to_thread(
                _slice_and_accumulate_sync,
                sorted_df,
                split_values,
                len(boundaries),
                col_names,
                col_types,
                idx_col_idx,
                context.br(),
                accumulated,
            )
            del sorted_df  # Free; sub-views are in accumulated as spill-eligible chunks

        # Sync pool streams so memory freed during Phase 2 is available
        for _ in range(context.stream_pool_size()):
            context.get_stream_from_pool().synchronize()

        # Phase 3: per-boundary output
        # TODO: if all input rows were shuffled to other ranks, accumulated is
        # all-empty while boundaries is non-empty (seq-num gap).  Revisit once
        # rapidsmpf empty-partition handling is clearer.
        if col_names is not None:
            assert col_types is not None
            for k, (seq_num, _start, _end) in enumerate(boundaries):
                # Bring sub-chunks to device if spilled (async wait).
                available: list[TableChunk] = []
                for chunk in accumulated[k]:
                    avail = await chunk.make_available_or_wait(
                        context, net_memory_delta=0
                    )
                    available.append(avail)
                accumulated[k] = []  # drop refs to now-consumed originals

                result = await asyncio.to_thread(
                    _concat_sort_boundary_sync,
                    available,
                    col_names,
                    col_types,
                    idx_col_idx,
                    keep_col_indices,
                    ir_context,
                    context.br(),
                )
                if tracer is not None:
                    tracer.add_chunk(table=result.table_view())
                await ch_out.send(context, Message(seq_num, result))
                del available

        await ch_out.drain(context)


@generate_ir_sub_network.register(Over)
def _(
    ir: Over, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    wrapped_ir = ir.children[0]
    actors, channels = process_children(wrapped_ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    actors[ir] = [
        over_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[wrapped_ir.children[0]].reserve_output_slot(),
            collective_ids.pop(),
        )
    ]
    return actors, channels
