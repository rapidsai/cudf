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
from cudf_polars.dsl.expr import Col, GroupedWindow, NamedExpr
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.ir import IR, GroupBy, HStack, Select
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.experimental.groupby import combine, decompose
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
from cudf_polars.experimental.utils import (
    _all_over_scalar_and_top_level,
    _extract_over_shuffle_indices,
)

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


def _build_over_groupby_irs(
    gw_nodes: list[GroupedWindow],
    child_ir: IR,
) -> tuple[GroupBy, GroupBy, Select | None]:
    """
    Build piecewise, reduction, and (optionally) selection GroupBy IRs.

    Parameters
    ----------
    gw_nodes
        Top-level GroupedWindow nodes sharing the same partition-by keys;
        all must be scalar (Agg/Len only in named_aggs).
    child_ir
        Child IR of the enclosing Select/HStack; defines the input schema.

    Returns
    -------
    piecewise_ir
        GroupBy IR that computes partial aggregates per chunk.
    reduction_ir
        GroupBy IR that reduces partial aggregates to a single result.
    select_ir
        Select IR for post-aggregation expressions (e.g. division for mean);
        None when all aggregations are pass-through.
    """
    gw = gw_nodes[0]
    by_exprs = [e for e in gw.children[: gw.by_count] if isinstance(e, Col)]
    key_named_exprs = [NamedExpr(e.name, e) for e in by_exprs]
    key_schema = {e.name: child_ir.schema[e.name] for e in by_exprs}

    seen: set[str] = set()
    all_scalar_named: list[NamedExpr] = []
    for gw_node in gw_nodes:
        reductions, _ = gw_node._split_named_expr()
        for ne in reductions:
            if ne.name not in seen:
                all_scalar_named.append(ne)
                seen.add(ne.name)

    name_gen = unique_names(child_ir.schema.keys())
    decompositions = [
        decompose(ne.name, ne.value, names=name_gen) for ne in all_scalar_named
    ]
    selection_exprs, piecewise_exprs, reduction_exprs, _ = combine(*decompositions)

    pwise_schema = dict(key_schema) | {
        ne.name: ne.value.dtype for ne in piecewise_exprs
    }
    piecewise_ir = GroupBy(
        pwise_schema,
        key_named_exprs,
        piecewise_exprs,
        False,  # noqa: FBT003
        None,
        child_ir,
    )

    reduction_key_exprs = [
        NamedExpr(ne.name, Col(pwise_schema[ne.name], ne.name))
        for ne in key_named_exprs
    ]
    reduction_schema = {ne.name: pwise_schema[ne.name] for ne in key_named_exprs} | {
        ne.name: ne.value.dtype for ne in reduction_exprs
    }
    reduction_ir = GroupBy(
        reduction_schema,
        reduction_key_exprs,
        reduction_exprs,
        False,  # noqa: FBT003
        None,
        piecewise_ir,
    )

    select_ir: Select | None
    if any(
        ne.name not in reduction_schema or reduction_schema[ne.name] != ne.value.dtype
        for ne in selection_exprs
    ):
        select_key_exprs = [
            NamedExpr(ne.name, Col(reduction_schema[ne.name], ne.name))
            for ne in key_named_exprs
        ]
        select_schema = {
            ne.name: reduction_schema[ne.name] for ne in key_named_exprs
        } | {ne.name: ne.value.dtype for ne in selection_exprs}
        select_ir = Select(
            select_schema,
            [*select_key_exprs, *selection_exprs],
            False,  # noqa: FBT003
            reduction_ir,
        )
    else:
        select_ir = None

    return piecewise_ir, reduction_ir, select_ir


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
    gw_nodes: list[GroupedWindow],
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
    gw_nodes: list[GroupedWindow],
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


def _sort_and_split_sync(
    dfs: list[DataFrame],
    row_idx_col: str,
    boundaries: list[tuple[int, int, int]],
    stream: Any,
    br: Any,
) -> list[tuple[int, TableChunk]]:
    """Sort all partition results by row_idx and split at original chunk boundaries."""
    if not dfs:
        return []

    col_names = [c.name for c in dfs[0].columns]
    merged = plc.concatenate.concatenate([df.table for df in dfs], stream=stream)

    idx_col_idx = col_names.index(row_idx_col)
    sort_order = plc.sorting.stable_sorted_order(
        plc.Table([merged.columns()[idx_col_idx]]),
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.AFTER],
        stream=stream,
    )

    keep_indices = [i for i, n in enumerate(col_names) if n != row_idx_col]
    sorted_tbl = plc.copying.gather(
        plc.Table([merged.columns()[i] for i in keep_indices]),
        sort_order,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )

    split_pts = [end for _, _, end in boundaries[:-1]]
    sub_tbls = (
        plc.copying.split(sorted_tbl, split_pts, stream=stream)
        if split_pts
        else [sorted_tbl]
    )

    return [
        (
            seq_num,
            TableChunk.from_pylibcudf_table(t, stream, exclusive_view=True, br=br),
        )
        for (seq_num, _, _), t in zip(boundaries, sub_tbls, strict=False)
    ]


@define_actor()
async def over_actor(
    context: Context,
    comm: Communicator,
    ir: Select | HStack,
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
        The Select or HStack IR node containing GroupedWindow expressions.
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
        metadata_in = await recv_metadata(ch_in, context)

        exprs = [e.value for e in (ir.exprs if isinstance(ir, Select) else ir.columns)]
        key_indices = _extract_over_shuffle_indices(exprs, ir.children[0].schema)
        assert key_indices is not None
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
                partitioning=maybe_remap_partitioning(ir, metadata_in.partitioning),
                duplicated=metadata_in.duplicated,
            )
            await chunkwise_evaluate(
                context, ir, ir_context, ch_out, ch_in, metadata_out, tracer=tracer
            )
            return

        if _all_over_scalar_and_top_level(exprs):
            gw_nodes = [e for e in exprs if isinstance(e, GroupedWindow)]
            key_names = tuple(
                e.name
                for e in gw_nodes[0].children[: gw_nodes[0].by_count]
                if isinstance(e, Col)
            )

            piecewise_ir, reduction_ir, select_ir = _build_over_groupby_irs(
                gw_nodes, ir.children[0]
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
            # all ranks' partial results; select_ir is applied only once after.
            if comm.nranks > 1 and not metadata_in.duplicated:
                allgather = AllGatherManager(context, comm, collective_id)
                allgather.insert(0, local_agg)
                allgather.insert_finished()
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

            if select_ir is not None:
                global_agg = await evaluate_chunk(
                    context, global_agg, select_ir, ir_context=ir_context
                )

            final_agg_ir = select_ir if select_ir is not None else reduction_ir
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
            return

        modulus = max(comm.nranks, metadata_in.local_count)
        metadata_out = ChannelMetadata(
            local_count=metadata_in.local_count,
            partitioning=maybe_remap_partitioning(ir, metadata_in.partitioning),
            duplicated=metadata_in.duplicated,
        )
        await send_metadata(ch_out, context, metadata_out)

        shuffle = ShuffleManager(context, comm, modulus, collective_id)
        row_counter = 0
        row_idx_col = next(unique_names(ir.children[0].schema.keys()))
        boundaries: list[tuple[int, int, int]] = []

        while (msg := await ch_in.recv(context)) is not None:
            chunk = TableChunk.from_message(
                msg, br=context.br()
            ).make_available_and_spill(context.br(), allow_overbooking=True)
            stream = ir_context.get_cuda_stream()
            n_rows = chunk.table_view().num_rows()
            boundaries.append((msg.sequence_number, row_counter, row_counter + n_rows))
            chunk = await asyncio.to_thread(
                _add_row_idx_sync, chunk, row_counter, stream, context.br()
            )
            row_counter += n_rows
            shuffle.insert_hash(chunk, key_indices)

        await shuffle.insert_finished()

        partition_results: list[DataFrame] = []
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
                ir,
                ir_context,
                row_idx_col,
            )
            partition_results.append(result_df)

        stream = ir_context.get_cuda_stream()
        for seq_num, result in await asyncio.to_thread(
            _sort_and_split_sync,
            partition_results,
            row_idx_col,
            boundaries,
            stream,
            context.br(),
        ):
            if tracer is not None:
                tracer.add_chunk(table=result.table_view())
            await ch_out.send(context, Message(seq_num, result))

        await ch_out.drain(context)


@generate_ir_sub_network.register(Select)
@generate_ir_sub_network.register(HStack)
def _(
    ir: Select | HStack, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]

    if config_options.executor.dynamic_planning is None:
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    exprs = [e.value for e in (ir.exprs if isinstance(ir, Select) else ir.columns)]
    indices = _extract_over_shuffle_indices(exprs, ir.children[0].schema)

    if not (indices is not None and len(indices) > 0):
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

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
            collective_ids.pop(),
        )
    ]
    return actors, channels
