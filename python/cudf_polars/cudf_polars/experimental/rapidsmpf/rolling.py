# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Distributed rolling window actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl.ir import Rolling
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    gather_in_task_group,
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

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


def _ordinal_to_native(type_id: plc.TypeId, ordinal: int) -> int:
    """Map Polars duration ordinals (ns) to native index units. See ``duration_to_scalar``."""
    if type_id in (plc.TypeId.INT64, plc.TypeId.TIMESTAMP_NANOSECONDS):
        return ordinal
    elif type_id == plc.TypeId.TIMESTAMP_MICROSECONDS:
        return ordinal // 1000
    elif type_id == plc.TypeId.TIMESTAMP_MILLISECONDS:
        return ordinal // 1_000_000
    elif type_id == plc.TypeId.TIMESTAMP_DAYS:
        return ordinal // 86_400_000_000_000
    else:  # pragma: no cover
        raise NotImplementedError(
            f"Unsupported index type {type_id!r} for rolling window halo exchange"
        )


def _get_idx_col_i64(
    table: plc.Table,
    index_col_idx: int,
    type_id: plc.TypeId,
    i64: plc.DataType,
    stream: Stream,
) -> plc.Column:
    col = table.columns()[index_col_idx]
    if type_id != plc.TypeId.INT64:
        col = plc.unary.cast(col, i64, stream=stream)
    return col


def _minmax_py(
    col_i64: plc.Column, i64: plc.DataType, stream: Stream
) -> tuple[int, int]:
    """Return (min, max) of an INT64 column as Python ints via a D→H transfer."""
    mn = plc.reduce.reduce(col_i64, plc.aggregation.min(), i64, stream=stream)
    mx = plc.reduce.reduce(col_i64, plc.aggregation.max(), i64, stream=stream)
    mn_val = mn.to_py(stream=stream)
    mx_val = mx.to_py(stream=stream)
    assert isinstance(mn_val, int)
    assert isinstance(mx_val, int)
    return mn_val, mx_val


def _filter_ge(
    table: plc.Table,
    idx_col_i64: plc.Column,
    threshold: int,
    i64: plc.DataType,
    bool8: plc.DataType,
    stream: Stream,
) -> plc.Table:
    thr = plc.Scalar.from_py(threshold, i64, stream=stream)
    mask = plc.binaryop.binary_operation(
        idx_col_i64,
        thr,
        plc.binaryop.BinaryOperator.GREATER_EQUAL,
        bool8,
        stream=stream,
    )
    return plc.stream_compaction.apply_boolean_mask(table, mask, stream=stream)


def _filter_le(
    table: plc.Table,
    idx_col_i64: plc.Column,
    threshold: int,
    i64: plc.DataType,
    bool8: plc.DataType,
    stream: Stream,
) -> plc.Table:
    thr = plc.Scalar.from_py(threshold, i64, stream=stream)
    mask = plc.binaryop.binary_operation(
        idx_col_i64, thr, plc.binaryop.BinaryOperator.LESS_EQUAL, bool8, stream=stream
    )
    return plc.stream_compaction.apply_boolean_mask(table, mask, stream=stream)


@dataclass
class _RollingActState:
    left_ctx_df: DataFrame | None = None
    right_halo_df: DataFrame | None = None


def _prepare_expanded_rolling_frame(
    state: _RollingActState,
    *,
    context: Context,
    cur_msg: Message,
    next_msg: Message | None,
    is_last_chunk: bool,
    ir_context: IRExecutionContext,
    index_col_idx: int,
    type_id: plc.TypeId,
    lookback: int,
    lookahead: int,
    i64: plc.DataType,
    bool8: plc.DataType,
    ext_col_names: list[str],
    ext_dtypes: list[Any],
) -> tuple[DataFrame, int, int, DataFrame, int] | None:
    left_ctx_df = state.left_ctx_df
    right_halo_df = state.right_halo_df
    br = context.br()

    chunk = TableChunk.from_message(cur_msg, br)
    chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
    chunk_table = chunk.table_view()
    n_chunk_rows = chunk_table.num_rows()
    if n_chunk_rows == 0:
        return None

    chunk_stream = chunk.stream
    need_chunk_minmax = (
        left_ctx_df is not None and left_ctx_df.num_rows > 0 and lookback > 0
    ) or (lookahead > 0 and not is_last_chunk)
    chunk_mn = 0
    chunk_mx = 0
    if need_chunk_minmax:
        chunk_idx = _get_idx_col_i64(
            chunk_table, index_col_idx, type_id, i64, chunk_stream
        )
        chunk_mn, chunk_mx = _minmax_py(chunk_idx, i64, chunk_stream)

    if left_ctx_df is not None and left_ctx_df.num_rows > 0 and lookback > 0:
        left_idx = _get_idx_col_i64(
            left_ctx_df.table, index_col_idx, type_id, i64, left_ctx_df.stream
        )
        left_ctx_df = DataFrame.from_table(
            _filter_ge(
                left_ctx_df.table,
                left_idx,
                chunk_mn - lookback,
                i64,
                bool8,
                left_ctx_df.stream,
            ),
            ext_col_names,
            ext_dtypes,
            stream=left_ctx_df.stream,
        )

    if lookahead > 0:
        if is_last_chunk:
            right_ctx_df = right_halo_df
        else:
            assert next_msg is not None
            next_chunk = TableChunk.from_message(next_msg, br)
            next_chunk = next_chunk.make_available_and_spill(br, allow_overbooking=True)
            next_table = next_chunk.table_view()
            next_stream = next_chunk.stream
            next_idx = _get_idx_col_i64(
                next_table, index_col_idx, type_id, i64, next_stream
            )
            right_ctx_table = _filter_le(
                next_table, next_idx, chunk_mx + lookahead, i64, bool8, next_stream
            )
            right_ctx_df = (
                DataFrame.from_table(
                    right_ctx_table, ext_col_names, ext_dtypes, stream=next_stream
                )
                if right_ctx_table.num_rows() > 0
                else None
            )
    else:
        right_ctx_df = right_halo_df if is_last_chunk else None

    chunk_df = DataFrame.from_table(
        chunk_table, ext_col_names, ext_dtypes, stream=chunk_stream
    )
    chunk_id_name = ext_col_names[-1]
    cid_col = chunk_df.column_map[chunk_id_name].obj
    cid_mn, cid_mx = _minmax_py(cid_col, i64, chunk_stream)
    assert cid_mn == cid_mx
    center_chunk_id = cid_mn
    n_left = left_ctx_df.num_rows if left_ctx_df is not None else 0
    dfs = [df for df in (left_ctx_df, chunk_df, right_ctx_df) if df is not None]
    if len(dfs) == 1:
        combined_df = dfs[0]
    else:
        with ir_context.stream_ordered_after(*dfs) as s:
            combined_df = DataFrame.from_table(
                plc.concatenate.concatenate([df.table for df in dfs], stream=s),
                ext_col_names,
                ext_dtypes,
                stream=s,
            )

    state.left_ctx_df = left_ctx_df
    return combined_df, n_left, n_chunk_rows, chunk_df, center_chunk_id


async def _rolling_do_evaluate(
    ir: Rolling,
    ir_context: IRExecutionContext,
    non_child_args_no_zlice: tuple[Any, ...],
    combined_df: DataFrame,
) -> DataFrame:
    """Run ``Rolling.do_evaluate`` in a worker thread."""
    return await asyncio.to_thread(
        Rolling.do_evaluate,
        *non_child_args_no_zlice,
        combined_df,
        context=ir_context,
    )


def _rolling_output_with_zlice(
    result_df: DataFrame,
    *,
    n_left: int,
    n_chunk_rows: int,
    n_processed: int,
    zlice: tuple[int, int | None] | None,
) -> DataFrame | None:
    chunk_result_df = result_df.slice((n_left, n_chunk_rows))
    local_result_df: DataFrame | None = chunk_result_df
    if zlice is not None:
        zlice_offset, zlice_length = zlice
        local_start = max(0, zlice_offset - n_processed)
        local_end = (
            n_chunk_rows
            if zlice_length is None
            else min(n_chunk_rows, zlice_offset + zlice_length - n_processed)
        )
        local_result_df = (
            chunk_result_df.slice((local_start, local_end - local_start))
            if local_start < local_end
            else None
        )
    return local_result_df


def _rolling_append_left_context(
    state: _RollingActState,
    chunk_df: DataFrame,
    *,
    lookback: int,
    ir_context: IRExecutionContext,
    col_names: list[str],
    col_dtypes: list[Any],
) -> None:
    if lookback <= 0:
        return
    left_ctx_df = state.left_ctx_df
    if left_ctx_df is None or left_ctx_df.num_rows == 0:
        state.left_ctx_df = chunk_df
    else:
        with ir_context.stream_ordered_after(left_ctx_df, chunk_df) as s:
            state.left_ctx_df = DataFrame.from_table(
                plc.concatenate.concatenate(
                    [left_ctx_df.table, chunk_df.table], stream=s
                ),
                col_names,
                col_dtypes,
                stream=s,
            )


def _build_center_row_mask(
    combined: DataFrame,
    *,
    chunk_id_name: str,
    emit_id_name: str,
    bool8: plc.DataType,
) -> Column:
    prov = combined.column_map[chunk_id_name]
    emit = combined.column_map[emit_id_name]
    mask_obj = plc.binaryop.binary_operation(
        prov.obj,
        emit.obj,
        plc.binaryop.BinaryOperator.EQUAL,
        bool8,
        stream=combined.stream,
    )
    return Column(mask_obj, dtype=DataType(pl.Boolean()), name=None)


def _emit_id_column(
    center_chunk_id: int,
    n_rows: int,
    *,
    name: str,
    stream: Stream,
    i64: plc.DataType,
) -> Column:
    return Column(
        plc.Column.from_scalar(
            plc.Scalar.from_py(center_chunk_id, i64, stream=stream),
            n_rows,
            stream=stream,
        ),
        dtype=DataType(pl.Int64()),
        name=name,
    )


async def prepare_rank_boundaries(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    base_col_names: list[str],
    base_dtypes: list[Any],
    chunk_id_name: str,
    chunk_id_dtype: DataType,
    collective_ids: list[int],
) -> None:
    """Single-rank passthrough with per-message chunk id; multi-rank TBD (SparseAlltoall)."""
    assert len(collective_ids) == 2
    _ = collective_ids
    i64 = plc.DataType(plc.TypeId.INT64)
    next_chunk_id = 0
    br = context.br()
    while True:
        msg = await ch_in.recv(context)
        if msg is None:
            await ch_out.drain(context)
            return
        chunk = TableChunk.from_message(msg, br)
        chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
        stream = chunk.stream
        tv = chunk.table_view()
        df = DataFrame.from_table(tv, base_col_names, base_dtypes, stream=stream)
        id_col = Column(
            plc.Column.from_scalar(
                plc.Scalar.from_py(next_chunk_id, i64, stream=stream),
                df.num_rows,
                stream=stream,
            ),
            dtype=chunk_id_dtype,
            name=chunk_id_name,
        )
        next_chunk_id += 1
        out_df = df.with_columns([id_col], stream=stream)
        out_chunk = TableChunk.from_pylibcudf_table(
            out_df.table, stream, exclusive_view=True, br=br
        )
        await ch_out.send(context, Message(0, out_chunk))


async def prepare_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    ir_context: IRExecutionContext,
    state: _RollingActState,
    append_done: asyncio.Queue[None],
    *,
    lookback: int,
    lookahead: int,
    index_col_idx: int,
    type_id: plc.TypeId,
    i64: plc.DataType,
    bool8: plc.DataType,
    ext_col_names: list[str],
    ext_dtypes: list[Any],
    emit_id_name: str,
) -> None:
    """Expand rolling context on the boundary-tagged stream and send combined frames."""
    pending: Message | None = None
    while True:
        cur_msg = pending if pending is not None else await ch_in.recv(context)
        pending = None
        if cur_msg is None:
            await ch_out.drain(context)
            return

        if lookahead > 0:
            next_msg = await ch_in.recv(context)
            is_last = next_msg is None
        else:
            next_msg = None
            peek_msg = await ch_in.recv(context)
            is_last = peek_msg is None

        prep = _prepare_expanded_rolling_frame(
            state,
            context=context,
            cur_msg=cur_msg,
            next_msg=next_msg,
            is_last_chunk=is_last,
            ir_context=ir_context,
            index_col_idx=index_col_idx,
            type_id=type_id,
            lookback=lookback,
            lookahead=lookahead,
            i64=i64,
            bool8=bool8,
            ext_col_names=ext_col_names,
            ext_dtypes=ext_dtypes,
        )
        if prep is None:
            if lookahead > 0:
                pending = next_msg
            else:
                pending = peek_msg
            continue

        combined_df, _n_left, _n_chunk_rows, _, center_chunk_id = prep
        emit_col = _emit_id_column(
            center_chunk_id,
            combined_df.num_rows,
            name=emit_id_name,
            stream=combined_df.stream,
            i64=i64,
        )
        combined_tagged = combined_df.with_columns(
            [emit_col], stream=combined_df.stream
        )
        out_chunk = TableChunk.from_pylibcudf_table(
            combined_tagged.table,
            combined_tagged.stream,
            exclusive_view=True,
            br=context.br(),
        )
        await ch_out.send(context, Message(0, out_chunk))
        await append_done.get()

        if lookahead > 0:
            if is_last:
                break
            pending = next_msg
        else:
            if is_last:
                break
            pending = peek_msg

    await ch_out.drain(context)


async def rolling_eval_and_send(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    ir: Rolling,
    ir_context: IRExecutionContext,
    state: _RollingActState,
    append_done: asyncio.Queue[None],
    tracer: Any,
    *,
    lookback: int,
    base_col_names: list[str],
    base_dtypes: list[Any],
    ext_col_names: list[str],
    ext_dtypes: list[Any],
    chunk_id_name: str,
    emit_id_name: str,
    bool8: plc.DataType,
) -> None:
    """Evaluate rolling, mask to owned rows, zlice, send, and advance left context."""
    non_child_args_no_zlice = ir._non_child_args[:-1] + (None,)
    internal_meta = frozenset({chunk_id_name, emit_id_name})
    n_processed = 0
    br = context.br()
    while True:
        msg = await ch_in.recv(context)
        if msg is None:
            await ch_out.drain(context)
            return
        chunk = TableChunk.from_message(msg, br)
        chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
        combined = DataFrame.from_table(
            chunk.table_view(),
            ext_col_names,
            ext_dtypes,
            stream=chunk.stream,
        )
        mask = _build_center_row_mask(
            combined,
            chunk_id_name=chunk_id_name,
            emit_id_name=emit_id_name,
            bool8=bool8,
        )
        combined_eval = combined.discard_columns(internal_meta)
        result_df = await _rolling_do_evaluate(
            ir, ir_context, non_child_args_no_zlice, combined_eval
        )
        chunk_result = result_df.filter(mask)
        n_chunk_rows = chunk_result.num_rows
        local_result_df = _rolling_output_with_zlice(
            chunk_result,
            n_left=0,
            n_chunk_rows=n_chunk_rows,
            n_processed=n_processed,
            zlice=ir.zlice,
        )
        n_processed += n_chunk_rows
        await _rolling_send_chunk_if_nonempty(context, ch_out, local_result_df, tracer)
        center_only = combined.filter(mask).select(base_col_names)
        _rolling_append_left_context(
            state,
            center_only,
            lookback=lookback,
            ir_context=ir_context,
            col_names=base_col_names,
            col_dtypes=base_dtypes,
        )
        await append_done.put(None)


async def _rolling_send_chunk_if_nonempty(
    context: Context,
    ch_out: Channel[TableChunk],
    local_result_df: DataFrame | None,
    tracer: Any,
) -> None:
    if local_result_df is None or local_result_df.num_rows == 0:
        return
    result_chunk = TableChunk.from_pylibcudf_table(
        local_result_df.table,
        local_result_df.stream,
        exclusive_view=True,
        br=context.br(),
    )
    if tracer is not None:
        tracer.add_chunk(table=result_chunk.table_view())
    await ch_out.send(context, Message(0, result_chunk))


@define_actor()
async def rolling_actor(
    context: Context,
    comm: Communicator,
    ir: Rolling,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    collective_ids: list[int],
) -> None:
    """Rolling pipeline: rank boundaries → chunk prep → eval+send (internal Channels)."""
    assert comm.nranks == 1
    ch_after_boundaries: Channel[TableChunk] = context.create_channel()
    ch_prep_to_eval: Channel[TableChunk] = context.create_channel()
    async with shutdown_on_error(
        context,
        ch_in,
        ch_out,
        ch_after_boundaries,
        ch_prep_to_eval,
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        metadata_in = await recv_metadata(ch_in, context)
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(local_count=metadata_in.local_count, partitioning=None),
        )

        base_col_names = list(ir.children[0].schema.keys())
        base_dtypes = list(ir.children[0].schema.values())
        name_gen = unique_names(ir.children[0].schema.keys())
        chunk_id_name = next(name_gen)
        emit_id_name = next(name_gen)
        chunk_id_dtype = DataType(pl.Int64())

        type_id = ir.index_dtype.id()
        preceding_native = _ordinal_to_native(type_id, ir.preceding_ordinal)
        following_native = _ordinal_to_native(type_id, ir.following_ordinal)
        lookback = max(0, -preceding_native)
        lookahead = max(0, preceding_native + following_native)
        index_col_idx = base_col_names.index(ir.index.name)
        i64 = plc.DataType(plc.TypeId.INT64)
        bool8 = plc.DataType(plc.TypeId.BOOL8)

        ext_col_names = [*base_col_names, chunk_id_name]
        ext_dtypes = [*base_dtypes, chunk_id_dtype]

        act_state = _RollingActState()
        append_done: asyncio.Queue[None] = asyncio.Queue()

        await gather_in_task_group(
            prepare_rank_boundaries(
                context,
                ch_in,
                ch_after_boundaries,
                base_col_names,
                base_dtypes,
                chunk_id_name,
                chunk_id_dtype,
                collective_ids,
            ),
            prepare_chunks(
                context,
                ch_after_boundaries,
                ch_prep_to_eval,
                ir_context,
                act_state,
                append_done,
                lookback=lookback,
                lookahead=lookahead,
                index_col_idx=index_col_idx,
                type_id=type_id,
                i64=i64,
                bool8=bool8,
                ext_col_names=ext_col_names,
                ext_dtypes=ext_dtypes,
                emit_id_name=emit_id_name,
            ),
            rolling_eval_and_send(
                context,
                ch_prep_to_eval,
                ch_out,
                ir,
                ir_context,
                act_state,
                append_done,
                tracer,
                lookback=lookback,
                base_col_names=base_col_names,
                base_dtypes=base_dtypes,
                ext_col_names=[*ext_col_names, emit_id_name],
                ext_dtypes=[*ext_dtypes, chunk_id_dtype],
                chunk_id_name=chunk_id_name,
                emit_id_name=emit_id_name,
                bool8=bool8,
            ),
        )


@generate_ir_sub_network.register(Rolling)
def _(
    ir: Rolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    comm = rec.state["comm"]
    nodes, channels = process_children(ir, rec)
    if comm.nranks > 1:
        raise NotImplementedError(
            "LazyFrame.rolling for multiple ranks needs SparseAlltoall "
            "(https://github.com/rapidsai/rapidsmpf/pull/959)."
        )
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    assert len(collective_ids) == 2, (
        f"Rolling requires 2 collective IDs, got {len(collective_ids)}"
    )
    ctx = rec.state["context"]
    nodes[ir] = [
        rolling_actor(
            ctx,
            comm,
            ir,
            rec.state["ir_context"],
            ch_in=channels[ir.children[0]].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            collective_ids=list(collective_ids),
        ),
    ]
    return nodes, channels
