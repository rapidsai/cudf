# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Distributed rolling window actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl.ir import Rolling
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


_ORDINAL_DIVISOR: dict[plc.TypeId, int] = {
    plc.TypeId.INT64: 1,
    plc.TypeId.TIMESTAMP_NANOSECONDS: 1,
    plc.TypeId.TIMESTAMP_MICROSECONDS: 1_000,
    plc.TypeId.TIMESTAMP_MILLISECONDS: 1_000_000,
    plc.TypeId.TIMESTAMP_DAYS: 86_400_000_000_000,
}


def _ordinal_to_native(type_id: plc.TypeId, ordinal: int) -> int:
    """Map Polars duration ordinals (ns) to native index units. See ``duration_to_scalar``."""
    try:
        return ordinal // _ORDINAL_DIVISOR[type_id]
    except KeyError:  # pragma: no cover
        raise NotImplementedError(
            f"Unsupported index type {type_id!r} for rolling window halo exchange"
        ) from None


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
    col: plc.Column, reduce_dtype: plc.DataType, stream: Stream
) -> tuple[int, int]:
    """Return (min, max) of ``col`` as Python ints (min/max reduce at ``reduce_dtype``)."""
    mn = plc.reduce.reduce(col, plc.aggregation.min(), reduce_dtype, stream=stream)
    mx = plc.reduce.reduce(col, plc.aggregation.max(), reduce_dtype, stream=stream)
    mn_val = mn.to_py(stream=stream)
    mx_val = mx.to_py(stream=stream)
    assert isinstance(mn_val, int)
    assert isinstance(mx_val, int)
    return mn_val, mx_val


def _filter_threshold(
    table: plc.Table,
    idx_col_i64: plc.Column,
    threshold: int,
    op: plc.binaryop.BinaryOperator,
    i64: plc.DataType,
    bool8: plc.DataType,
    stream: Stream,
) -> plc.Table:
    thr = plc.Scalar.from_py(threshold, i64, stream=stream)
    mask = plc.binaryop.binary_operation(idx_col_i64, thr, op, bool8, stream=stream)
    return plc.stream_compaction.apply_boolean_mask(table, mask, stream=stream)


@dataclass
class _RollingActState:
    left_ctx_df: DataFrame | None = None
    right_halo_df: DataFrame | None = None


@dataclass(frozen=True)
class _RollingStreamChunkMeta:
    """Per-chunk metadata after rank-boundary tagging (internal channel 1 → 2)."""

    is_local_rank_chunk: bool


@dataclass(frozen=True)
class _RollingExpandedChunkMeta:
    """Per-chunk metadata for expanded frames (internal channel 2 → 3)."""

    center_begin: int
    center_end: int


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
    base_col_names: list[str],
    base_dtypes: list[Any],
    is_local_rank_chunk: bool,
) -> tuple[DataFrame, int, int] | None:
    if not is_local_rank_chunk:
        raise NotImplementedError(
            "Non-local rank boundary chunks are not supported for rolling yet."
        )

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
            _filter_threshold(
                left_ctx_df.table,
                left_idx,
                chunk_mn - lookback,
                plc.binaryop.BinaryOperator.GREATER_EQUAL,
                i64,
                bool8,
                left_ctx_df.stream,
            ),
            base_col_names,
            base_dtypes,
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
            right_ctx_table = _filter_threshold(
                next_table,
                next_idx,
                chunk_mx + lookahead,
                plc.binaryop.BinaryOperator.LESS_EQUAL,
                i64,
                bool8,
                next_stream,
            )
            right_ctx_df = (
                DataFrame.from_table(
                    right_ctx_table, base_col_names, base_dtypes, stream=next_stream
                )
                if right_ctx_table.num_rows() > 0
                else None
            )
    else:
        right_ctx_df = right_halo_df if is_last_chunk else None

    chunk_df = DataFrame.from_table(
        chunk_table, base_col_names, base_dtypes, stream=chunk_stream
    )
    n_left = left_ctx_df.num_rows if left_ctx_df is not None else 0
    dfs = [df for df in (left_ctx_df, chunk_df, right_ctx_df) if df is not None]
    if len(dfs) == 1:
        combined_df = dfs[0]
    else:
        with ir_context.stream_ordered_after(*dfs) as s:
            combined_df = DataFrame.from_table(
                plc.concatenate.concatenate([df.table for df in dfs], stream=s),
                base_col_names,
                base_dtypes,
                stream=s,
            )

    state.left_ctx_df = (
        left_ctx_df  # trim only; append happens in rolling_eval_and_send
    )
    return combined_df, n_left, n_chunk_rows


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
    n_chunk_rows: int,
    n_processed: int,
    zlice: tuple[int, int | None] | None,
) -> DataFrame | None:
    if zlice is None:
        return result_df
    zlice_offset, zlice_length = zlice
    local_start = max(0, zlice_offset - n_processed)
    local_end = (
        n_chunk_rows
        if zlice_length is None
        else min(n_chunk_rows, zlice_offset + zlice_length - n_processed)
    )
    return (
        result_df.slice((local_start, local_end - local_start))
        if local_start < local_end
        else None
    )


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


def _build_center_row_mask_from_range(
    n_rows: int,
    center_begin: int,
    center_end: int,
    bool8: plc.DataType,
    stream: Stream,
) -> Column:
    """Boolean mask with True on ``[center_begin, center_end)`` (half-open row indices)."""
    row_id = plc.filling.sequence(
        n_rows,
        plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream),
        plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=stream),
        stream=stream,
    )
    ge = plc.binaryop.binary_operation(
        row_id,
        plc.Scalar.from_py(center_begin, plc.types.SIZE_TYPE, stream=stream),
        plc.binaryop.BinaryOperator.GREATER_EQUAL,
        bool8,
        stream=stream,
    )
    lt = plc.binaryop.binary_operation(
        row_id,
        plc.Scalar.from_py(center_end, plc.types.SIZE_TYPE, stream=stream),
        plc.binaryop.BinaryOperator.LESS,
        bool8,
        stream=stream,
    )
    mask_obj = plc.binaryop.binary_operation(
        ge, lt, plc.binaryop.BinaryOperator.LOGICAL_AND, bool8, stream=stream
    )
    return Column(mask_obj, dtype=DataType(pl.Boolean()), name=None)


async def _drain_data_messages(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
) -> None:
    """Ensure all data messages are drained."""
    remaining_msg = await ch_in.recv(context)
    if remaining_msg is not None:
        raise RuntimeError("Expected all messages to be drained.")
    await ch_out.drain(context)


async def prepare_rank_boundaries(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    collective_ids: list[int],
) -> None:
    """
    Relay chunks with paired per-chunk metadata (single-rank: always local).

    Multi-rank will extend this stage to inject rank-boundary frames with
    ``is_local_rank_chunk=False`` without changing downstream contracts.
    """
    assert len(collective_ids) == 2
    _ = collective_ids
    br = context.br()
    while (msg := await ch_in.recv(context)) is not None:
        seq_num = msg.sequence_number
        chunk = TableChunk.from_message(msg, br)
        chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
        stream = chunk.stream
        await ch_out.send_metadata(
            context,
            Message(
                seq_num,
                ArbitraryChunk(_RollingStreamChunkMeta(is_local_rank_chunk=True)),
            ),
        )
        out_chunk = TableChunk.from_pylibcudf_table(
            chunk.table_view(), stream, exclusive_view=True, br=br
        )
        await ch_out.send(context, Message(seq_num, out_chunk))

    await ch_out.drain_metadata(context)
    await ch_out.drain(context)


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
    base_col_names: list[str],
    base_dtypes: list[Any],
) -> None:
    """
    Expand rolling context; each output uses paired metadata + data messages.

    Metadata carries the center row range in the combined table; the data
    :class:`Message` keeps the upstream ``sequence_number`` for the center chunk.
    """
    pending: tuple[_RollingStreamChunkMeta, Message] | None = None
    while True:
        if pending is not None:
            cur_meta, cur_msg = pending
            pending = None
        else:
            if (cur_meta_msg := await ch_in.recv_metadata(context)) is None:
                break
            cur_meta = ArbitraryChunk.from_message(cur_meta_msg).release()
            assert isinstance(cur_meta, _RollingStreamChunkMeta)
            cur_msg = await ch_in.recv(context)
            assert cur_msg is not None, "Expected data message after metadata."

        next_meta: _RollingStreamChunkMeta | None = None
        next_msg: Message | None = None
        peek_meta_msg = await ch_in.recv_metadata(context)
        if peek_meta_msg is None:
            is_last = True
        else:
            next_meta = ArbitraryChunk.from_message(peek_meta_msg).release()
            assert isinstance(next_meta, _RollingStreamChunkMeta)
            next_msg = await ch_in.recv(context)
            is_last = next_msg is None

        prep = _prepare_expanded_rolling_frame(
            state,
            context=context,
            cur_msg=cur_msg,
            next_msg=next_msg if lookahead > 0 else None,
            is_last_chunk=is_last,
            ir_context=ir_context,
            index_col_idx=index_col_idx,
            type_id=type_id,
            lookback=lookback,
            lookahead=lookahead,
            i64=i64,
            bool8=bool8,
            base_col_names=base_col_names,
            base_dtypes=base_dtypes,
            is_local_rank_chunk=cur_meta.is_local_rank_chunk,
        )
        if prep is None:
            if next_msg is not None:
                assert next_meta is not None
                pending = (next_meta, next_msg)
            continue

        combined_df, n_left, n_chunk_rows = prep
        cur_seq = cur_msg.sequence_number
        expanded_meta = _RollingExpandedChunkMeta(
            center_begin=n_left,
            center_end=n_left + n_chunk_rows,
        )
        await ch_out.send_metadata(
            context,
            Message(cur_seq, ArbitraryChunk(expanded_meta)),
        )
        out_chunk = TableChunk.from_pylibcudf_table(
            combined_df.table,
            combined_df.stream,
            exclusive_view=True,
            br=context.br(),
        )
        await ch_out.send(context, Message(cur_seq, out_chunk))
        await append_done.get()

        if is_last:
            break
        if next_msg is not None:
            assert next_meta is not None
            pending = (next_meta, next_msg)

    await _drain_data_messages(context, ch_in, ch_out)
    await ch_out.drain_metadata(context)


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
    bool8: plc.DataType,
) -> None:
    """Evaluate rolling, mask to owned rows, zlice, send, and advance left context."""
    non_child_args_no_zlice = ir._non_child_args[:-1] + (None,)
    n_processed = 0
    br = context.br()
    while (meta_msg := await ch_in.recv_metadata(context)) is not None:
        expanded = ArbitraryChunk.from_message(meta_msg).release()
        assert isinstance(expanded, _RollingExpandedChunkMeta)
        msg = await ch_in.recv(context)
        assert msg is not None, "Expected data message after metadata."
        seq_num = msg.sequence_number
        chunk = TableChunk.from_message(msg, br)
        chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
        combined = DataFrame.from_table(
            chunk.table_view(),
            base_col_names,
            base_dtypes,
            stream=chunk.stream,
        )
        mask = _build_center_row_mask_from_range(
            combined.num_rows,
            expanded.center_begin,
            expanded.center_end,
            bool8,
            combined.stream,
        )
        result_df = await _rolling_do_evaluate(
            ir, ir_context, non_child_args_no_zlice, combined
        )
        chunk_result = result_df.filter(mask)
        n_chunk_rows = chunk_result.num_rows
        local_result_df = _rolling_output_with_zlice(
            chunk_result,
            n_chunk_rows=n_chunk_rows,
            n_processed=n_processed,
            zlice=ir.zlice,
        )
        n_processed += n_chunk_rows
        await _rolling_send_chunk_if_nonempty(
            context, ch_out, local_result_df, tracer, sequence_number=seq_num
        )
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

    await _drain_data_messages(context, ch_in, ch_out)


async def _rolling_send_chunk_if_nonempty(
    context: Context,
    ch_out: Channel[TableChunk],
    local_result_df: DataFrame | None,
    tracer: Any,
    *,
    sequence_number: int,
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
    await ch_out.send(context, Message(sequence_number, result_chunk))


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

        type_id = ir.index_dtype.id()
        preceding_native = _ordinal_to_native(type_id, ir.preceding_ordinal)
        following_native = _ordinal_to_native(type_id, ir.following_ordinal)
        lookback = max(0, -preceding_native)
        lookahead = max(0, preceding_native + following_native)
        index_col_idx = base_col_names.index(ir.index.name)
        i64 = plc.DataType(plc.TypeId.INT64)
        bool8 = plc.DataType(plc.TypeId.BOOL8)

        act_state = _RollingActState()
        append_done: asyncio.Queue[None] = asyncio.Queue()

        await gather_in_task_group(
            prepare_rank_boundaries(
                context,
                ch_in,
                ch_after_boundaries,
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
                base_col_names=base_col_names,
                base_dtypes=base_dtypes,
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
