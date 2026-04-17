# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Distributed rolling window actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
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
from cudf_polars.dsl.utils.windows import rolling_stream_halo_extents
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
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


_TIMESTAMP_TO_DURATION: dict[plc.TypeId, plc.TypeId] = {
    plc.TypeId.TIMESTAMP_NANOSECONDS: plc.TypeId.DURATION_NANOSECONDS,
    plc.TypeId.TIMESTAMP_MICROSECONDS: plc.TypeId.DURATION_MICROSECONDS,
    plc.TypeId.TIMESTAMP_MILLISECONDS: plc.TypeId.DURATION_MILLISECONDS,
    plc.TypeId.TIMESTAMP_DAYS: plc.TypeId.DURATION_DAYS,
}

_INT64 = plc.DataType(plc.TypeId.INT64)
_BOOL8 = plc.DataType(plc.TypeId.BOOL8)


def _duration_dtype_for_timestamp(index_dtype: plc.DataType) -> plc.DataType:
    try:
        return plc.DataType(_TIMESTAMP_TO_DURATION[index_dtype.id()])
    except KeyError as e:  # pragma: no cover
        raise NotImplementedError(
            f"Unsupported timestamp index type {index_dtype!r} for rolling halo"
        ) from e


def _scalar_binop_scalar(
    lhs: plc.Scalar,
    rhs: plc.Scalar,
    op: plc.binaryop.BinaryOperator,
    out_dtype: plc.DataType,
    stream: Stream,
) -> plc.Scalar:
    """Apply ``op`` to two scalars (via 1-row columns); result dtype ``out_dtype``."""
    lc = plc.Column.from_scalar(lhs, 1, stream=stream)
    rc = plc.Column.from_scalar(rhs, 1, stream=stream)
    out = plc.binaryop.binary_operation(lc, rc, op, out_dtype, stream=stream)
    return out.to_scalar(stream=stream)


def _get_idx_col(
    table: plc.Table,
    index_col_idx: int,
    index_dtype: plc.DataType,
    stream: Stream,
) -> plc.Column:
    col = table.columns()[index_col_idx]
    if index_dtype.id() == plc.TypeId.INT64 and col.type().id() != plc.TypeId.INT64:
        col = plc.unary.cast(col, _INT64, stream=stream)
    return col


def _minmax_py(
    col: plc.Column,
    reduce_dtype: plc.DataType,
    stream: Stream,
    *,
    assume_sorted: bool = False,
) -> tuple[int, int]:
    """
    Return (min, max) of ``col`` as Python ints (``reduce_dtype`` for reduce path).

    If ``assume_sorted`` is True, use first/last rows (caller must guarantee non-decreasing
    order and no nulls, as for Polars rolling index columns).
    """
    if assume_sorted:
        n = col.size()
        lo_c = plc.copying.slice(col, [0, 1], stream=stream)[0]
        hi_c = plc.copying.slice(col, [n - 1, n], stream=stream)[0]
        mn_val = lo_c.to_scalar(stream=stream).to_py(stream=stream)
        mx_val = hi_c.to_scalar(stream=stream).to_py(stream=stream)
    else:
        mn_val = plc.reduce.reduce(
            col, plc.aggregation.min(), reduce_dtype, stream=stream
        ).to_py(stream=stream)
        mx_val = plc.reduce.reduce(
            col, plc.aggregation.max(), reduce_dtype, stream=stream
        ).to_py(stream=stream)
    assert isinstance(mn_val, int)
    assert isinstance(mx_val, int)
    return mn_val, mx_val


def _minmax_scalars(
    col: plc.Column,
    value_dtype: plc.DataType,
    stream: Stream,
    *,
    assume_sorted: bool = False,
) -> tuple[plc.Scalar, plc.Scalar]:
    if assume_sorted:
        n = col.size()
        lo_c = plc.copying.slice(col, [0, 1], stream=stream)[0]
        hi_c = plc.copying.slice(col, [n - 1, n], stream=stream)[0]
        return lo_c.to_scalar(stream=stream), hi_c.to_scalar(stream=stream)
    mn = plc.reduce.reduce(col, plc.aggregation.min(), value_dtype, stream=stream)
    mx = plc.reduce.reduce(col, plc.aggregation.max(), value_dtype, stream=stream)
    return mn, mx


def _filter_threshold(
    table: plc.Table,
    idx_col: plc.Column,
    threshold: int | plc.Scalar,
    op: plc.binaryop.BinaryOperator,
    stream: Stream,
) -> plc.Table:
    thr = (
        threshold
        if isinstance(threshold, plc.Scalar)
        else plc.Scalar.from_py(threshold, _INT64, stream=stream)
    )
    mask = plc.binaryop.binary_operation(idx_col, thr, op, _BOOL8, stream=stream)
    return plc.stream_compaction.apply_boolean_mask(table, mask, stream=stream)


@dataclass
class _RollingActState:
    left_ctx_df: DataFrame | None = None
    # Reserved for cross-rank streaming: right halo rows received from the next rank
    # when ``is_last_chunk`` is false locally but the global partition continues.
    # Single-rank / local-only pipelines never set this; it stays ``None``.
    right_halo_df: DataFrame | None = None
    prev_ungrouped_int_max: int | None = (
        None  # ungrouped INT64: max index on prior centers
    )
    # Chunks already received from the stream that are not the current center yet
    # (ordered); used to supply a right halo that may span multiple partitions.
    prefetch: deque[tuple[_RollingStreamChunkMeta, Message]] = field(
        default_factory=deque,
        repr=False,
    )


def _check_ungrouped_int_chunk_order(
    st: _RollingActState,
    combined: DataFrame,
    exp: _RollingExpandedChunkMeta,
    ir: Rolling,
    *,
    index_col_idx: int,
    seq: int,
) -> None:
    """Ungrouped INT64 only: center min must be >= prior center max (partition order)."""
    if ir.keys or ir.index_dtype.id() != plc.TypeId.INT64:
        return
    if (n := exp.center_end - exp.center_begin) == 0:
        return
    idx = _get_idx_col(
        combined.slice((exp.center_begin, n)).table,
        index_col_idx,
        ir.index_dtype,
        combined.stream,
    )
    lo, hi = _minmax_py(idx, _INT64, combined.stream, assume_sorted=True)
    if st.prev_ungrouped_int_max is not None and lo < st.prev_ungrouped_int_max:
        raise RuntimeError(
            f"rolling streaming: INT64 index decreased across chunks "
            f"(min {lo} < prev max {st.prev_ungrouped_int_max}, seq={seq})"
        )  # pragma: no cover; Should never get here
    st.prev_ungrouped_int_max = hi


@dataclass(frozen=True)
class _RollingStreamChunkMeta:
    """Per-chunk metadata after rank-boundary tagging (internal channel 1 → 2)."""

    is_local_rank_chunk: bool


@dataclass(frozen=True)
class _RollingExpandedChunkMeta:
    """Per-chunk metadata for expanded frames (internal channel 2 → 3)."""

    center_begin: int
    center_end: int


def _message_peek_copy(msg: Message, br: Any) -> Message:
    """
    Deep-copy a table message so ``TableChunk.from_message`` can consume the copy.

    ``TableChunk.from_message`` moves the payload out of the message (it becomes empty);
    use this for index-bounds / row-count probes that must not invalidate messages still
    queued for ``_prepare_expanded_rolling_frame``.
    """
    res = br.reserve_device_memory_and_spill(msg.copy_cost(), allow_overbooking=True)
    return msg.copy(res)


def _chunk_index_int_bounds_from_msg(
    msg: Message,
    br: Any,
    *,
    index_col_idx: int,
    index_dtype: plc.DataType,
) -> tuple[int, int] | None:
    """Return (min, max) index for a non-empty chunk, else None (consumes ``msg``)."""
    chunk = TableChunk.from_message(msg, br)
    chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
    table = chunk.table_view()
    if table.num_rows() == 0:
        return None
    stream = chunk.stream
    idx = _get_idx_col(table, index_col_idx, index_dtype, stream)
    return _minmax_py(idx, _INT64, stream, assume_sorted=True)


def _chunk_index_ts_bounds_from_msg(
    msg: Message,
    br: Any,
    *,
    index_col_idx: int,
    index_dtype: plc.DataType,
) -> tuple[plc.Scalar, plc.Scalar] | None:
    """Return (min, max) index scalars (consumes ``msg``)."""
    chunk = TableChunk.from_message(msg, br)
    chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
    table = chunk.table_view()
    if table.num_rows() == 0:
        return None
    stream = chunk.stream
    idx = _get_idx_col(table, index_col_idx, index_dtype, stream)
    return _minmax_scalars(idx, index_dtype, stream, assume_sorted=True)


def _int_right_halo_covers_prefetch(
    prefetch: deque[tuple[_RollingStreamChunkMeta, Message]],
    br: Any,
    *,
    index_col_idx: int,
    index_dtype: plc.DataType,
    chunk_mx: int,
    lookahead: int,
) -> bool:
    """True if the last prefetched chunk's max index reaches ``chunk_mx + lookahead``."""
    if not prefetch:
        return False
    _, last_msg = prefetch[-1]
    bounds = _chunk_index_int_bounds_from_msg(
        _message_peek_copy(last_msg, br),
        br,
        index_col_idx=index_col_idx,
        index_dtype=index_dtype,
    )
    assert bounds is not None
    return bounds[1] >= chunk_mx + lookahead


async def _recv_stream_chunk_pair(
    context: Context,
    ch_in: Channel[TableChunk],
) -> tuple[_RollingStreamChunkMeta, Message] | None:
    """Receive one (metadata, data) pair from ``ch_in``, or None if the stream ended."""
    if (meta_msg := await ch_in.recv_metadata(context)) is None:
        return None
    meta = ArbitraryChunk.from_message(meta_msg).release()
    assert isinstance(meta, _RollingStreamChunkMeta)
    msg = await ch_in.recv(context)
    assert msg is not None, "Expected data message after metadata."
    return (meta, msg)


async def _recv_one_into_prefetch(
    context: Context,
    ch_in: Channel[TableChunk],
    state: _RollingActState,
) -> bool:
    """Receive one (metadata, data) pair into ``state.prefetch``; False if stream ended."""
    pair = await _recv_stream_chunk_pair(context, ch_in)
    if pair is None:
        return False
    state.prefetch.append(pair)
    return True


def _peek_n_rows_from_message(msg: Message, br: Any) -> int:
    """Row count without consuming ``msg`` (``from_message`` clears the payload)."""
    work = _message_peek_copy(msg, br)
    chunk = TableChunk.from_message(work, br)
    chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
    return chunk.table_view().num_rows()


async def _extend_prefetch_for_int_lookahead(
    context: Context,
    ch_in: Channel[TableChunk],
    state: _RollingActState,
    *,
    chunk_mx: int,
    lookahead: int,
    index_col_idx: int,
    index_dtype: plc.DataType,
) -> None:
    """Pull successor chunks until the right halo in index space is covered or EOS."""
    br = context.br()
    threshold = chunk_mx + lookahead
    if not state.prefetch and not await _recv_one_into_prefetch(context, ch_in, state):
        return
    first_bounds = _chunk_index_int_bounds_from_msg(
        _message_peek_copy(state.prefetch[0][1], br),
        br,
        index_col_idx=index_col_idx,
        index_dtype=index_dtype,
    )
    assert first_bounds is not None, "prefetch successor chunk must be non-empty"
    mn0, _ = first_bounds
    if mn0 > threshold:
        return
    while True:
        if _int_right_halo_covers_prefetch(
            state.prefetch,
            br,
            index_col_idx=index_col_idx,
            index_dtype=index_dtype,
            chunk_mx=chunk_mx,
            lookahead=lookahead,
        ):
            return
        if not await _recv_one_into_prefetch(context, ch_in, state):
            return


async def _extend_prefetch_for_ts_lookahead(
    context: Context,
    ch_in: Channel[TableChunk],
    state: _RollingActState,
    *,
    chunk_mx_s: plc.Scalar,
    lookahead: int,
    index_dtype: plc.DataType,
    index_col_idx: int,
    cur_stream: Any,
) -> None:
    """Timestamp index: extend prefetch until last chunk max reaches ``chunk_mx + lookahead``."""
    br = context.br()
    dur_dt = _duration_dtype_for_timestamp(index_dtype)
    lookahead_s = plc.Scalar.from_py(lookahead, dur_dt, stream=cur_stream)
    upper_s = _scalar_binop_scalar(
        chunk_mx_s,
        lookahead_s,
        plc.binaryop.BinaryOperator.ADD,
        index_dtype,
        cur_stream,
    )
    if not state.prefetch and not await _recv_one_into_prefetch(context, ch_in, state):
        return
    b0 = _chunk_index_ts_bounds_from_msg(
        _message_peek_copy(state.prefetch[0][1], br),
        br,
        index_col_idx=index_col_idx,
        index_dtype=index_dtype,
    )
    assert b0 is not None, "prefetch successor chunk must be non-empty"
    mn0_s, _ = b0
    if _scalar_binop_scalar(
        mn0_s,
        upper_s,
        plc.binaryop.BinaryOperator.GREATER,
        _BOOL8,
        cur_stream,
    ).to_py(stream=cur_stream):
        return
    while True:
        _, last_msg = state.prefetch[-1]
        last_bounds = _chunk_index_ts_bounds_from_msg(
            _message_peek_copy(last_msg, br),
            br,
            index_col_idx=index_col_idx,
            index_dtype=index_dtype,
        )
        assert last_bounds is not None
        _, last_mx_s = last_bounds
        if _scalar_binop_scalar(
            last_mx_s,
            upper_s,
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            _BOOL8,
            cur_stream,
        ).to_py(stream=cur_stream):
            return
        if not await _recv_one_into_prefetch(context, ch_in, state):
            return


async def _ensure_successor_prefetched(
    context: Context,
    ch_in: Channel[TableChunk],
    state: _RollingActState,
) -> None:
    """If there may be a chunk after the current center, ensure one is buffered (lookahead==0)."""
    if state.prefetch:
        return
    await _recv_one_into_prefetch(context, ch_in, state)


async def _ensure_right_halo_prefetched(
    context: Context,
    ch_in: Channel[TableChunk],
    state: _RollingActState,
    cur_msg: Message,
    br: Any,
    *,
    is_int_index: bool,
    index_col_idx: int,
    index_dtype: plc.DataType,
    lookahead: int,
) -> bool:
    """Extend prefetch until the right halo for ``cur_msg`` is covered. Returns False if empty."""
    if is_int_index:
        cur_bounds = _chunk_index_int_bounds_from_msg(
            _message_peek_copy(cur_msg, br),
            br,
            index_col_idx=index_col_idx,
            index_dtype=index_dtype,
        )
        if cur_bounds is None:
            return False
        await _extend_prefetch_for_int_lookahead(
            context,
            ch_in,
            state,
            chunk_mx=cur_bounds[1],
            lookahead=lookahead,
            index_col_idx=index_col_idx,
            index_dtype=index_dtype,
        )
    else:
        cur_peek_chunk = TableChunk.from_message(_message_peek_copy(cur_msg, br), br)
        cur_peek_chunk = cur_peek_chunk.make_available_and_spill(
            br, allow_overbooking=True
        )
        peek_table = cur_peek_chunk.table_view()
        if peek_table.num_rows() == 0:
            return False
        peek_stream = cur_peek_chunk.stream
        peek_idx = _get_idx_col(peek_table, index_col_idx, index_dtype, peek_stream)
        _, chunk_mx_s = _minmax_scalars(
            peek_idx, index_dtype, peek_stream, assume_sorted=True
        )
        await _extend_prefetch_for_ts_lookahead(
            context,
            ch_in,
            state,
            chunk_mx_s=chunk_mx_s,
            lookahead=lookahead,
            index_dtype=index_dtype,
            index_col_idx=index_col_idx,
            cur_stream=peek_stream,
        )
    return True


def _prepare_expanded_rolling_frame(
    state: _RollingActState,
    *,
    context: Context,
    cur_msg: Message,
    next_msgs: Sequence[Message] | None,
    is_last_chunk: bool,
    ir_context: IRExecutionContext,
    index_col_idx: int,
    index_dtype: plc.DataType,
    lookback: int,
    lookahead: int,
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
    chunk_df = DataFrame.from_table(
        chunk_table, base_col_names, base_dtypes, stream=chunk_stream
    )
    is_int_index = index_dtype.id() == plc.TypeId.INT64

    # Chunk index min/max are only needed to trim halo rows by the rolling window
    # in index-value space (see `rolling_stream_halo_extents` for `lookback` /
    # `lookahead`).  Left: drop rows in `left_ctx_df` below `chunk_mn - lookback`.
    # Right: on a following chunk, drop rows above `chunk_mx + lookahead` (no next
    # chunk on the last partition).
    need_chunk_minmax_for_left = (
        left_ctx_df is not None and left_ctx_df.num_rows > 0 and lookback > 0
    )
    need_chunk_minmax_for_right = lookahead > 0 and not is_last_chunk
    need_chunk_minmax = need_chunk_minmax_for_left or need_chunk_minmax_for_right
    chunk_mn, chunk_mx = 0, 0
    chunk_mn_s: plc.Scalar | None = None
    chunk_mx_s: plc.Scalar | None = None
    dur_dt: plc.DataType | None = None
    if need_chunk_minmax:
        chunk_idx = _get_idx_col(chunk_table, index_col_idx, index_dtype, chunk_stream)
        if is_int_index:
            chunk_mn, chunk_mx = _minmax_py(
                chunk_idx, _INT64, chunk_stream, assume_sorted=True
            )
        else:
            dur_dt = _duration_dtype_for_timestamp(index_dtype)
            chunk_mn_s, chunk_mx_s = _minmax_scalars(
                chunk_idx, index_dtype, chunk_stream, assume_sorted=True
            )

    if left_ctx_df is not None and left_ctx_df.num_rows > 0 and lookback > 0:
        left_idx = _get_idx_col(
            left_ctx_df.table, index_col_idx, index_dtype, left_ctx_df.stream
        )
        if is_int_index:
            left_ctx_df = DataFrame.from_table(
                _filter_threshold(
                    left_ctx_df.table,
                    left_idx,
                    chunk_mn - lookback,
                    plc.binaryop.BinaryOperator.GREATER_EQUAL,
                    left_ctx_df.stream,
                ),
                base_col_names,
                base_dtypes,
                stream=left_ctx_df.stream,
            )
        else:
            assert chunk_mn_s is not None
            assert dur_dt is not None
            lookback_s = plc.Scalar.from_py(lookback, dur_dt, stream=chunk_stream)
            lower_s = _scalar_binop_scalar(
                chunk_mn_s,
                lookback_s,
                plc.binaryop.BinaryOperator.SUB,
                index_dtype,
                chunk_stream,
            )
            with ir_context.stream_ordered_after(left_ctx_df, chunk_df) as s:
                left_ctx_df = DataFrame.from_table(
                    _filter_threshold(
                        left_ctx_df.table,
                        left_idx,
                        lower_s,
                        plc.binaryop.BinaryOperator.GREATER_EQUAL,
                        s,
                    ),
                    base_col_names,
                    base_dtypes,
                    stream=s,
                )

    if lookahead > 0:
        if is_last_chunk:
            right_ctx_df = right_halo_df
        else:
            assert next_msgs is not None
            right_tables_filtered: list[DataFrame] = []
            for next_msg in next_msgs:
                next_chunk = TableChunk.from_message(next_msg, br)
                next_chunk = next_chunk.make_available_and_spill(
                    br, allow_overbooking=True
                )
                next_table = next_chunk.table_view()
                next_stream = next_chunk.stream
                next_df = DataFrame.from_table(
                    next_table, base_col_names, base_dtypes, stream=next_stream
                )
                next_idx = _get_idx_col(
                    next_table, index_col_idx, index_dtype, next_stream
                )
                if is_int_index:
                    right_ctx_table = _filter_threshold(
                        next_table,
                        next_idx,
                        chunk_mx + lookahead,
                        plc.binaryop.BinaryOperator.LESS_EQUAL,
                        next_stream,
                    )
                    if right_ctx_table.num_rows() > 0:
                        right_tables_filtered.append(
                            DataFrame.from_table(
                                right_ctx_table,
                                base_col_names,
                                base_dtypes,
                                stream=next_stream,
                            )
                        )
                else:
                    assert chunk_mx_s is not None
                    assert dur_dt is not None
                    lookahead_s = plc.Scalar.from_py(
                        lookahead, dur_dt, stream=chunk_stream
                    )
                    upper_s = _scalar_binop_scalar(
                        chunk_mx_s,
                        lookahead_s,
                        plc.binaryop.BinaryOperator.ADD,
                        index_dtype,
                        chunk_stream,
                    )
                    with ir_context.stream_ordered_after(next_df, chunk_df) as s:
                        right_ctx_table = _filter_threshold(
                            next_table,
                            next_idx,
                            upper_s,
                            plc.binaryop.BinaryOperator.LESS_EQUAL,
                            s,
                        )
                        if right_ctx_table.num_rows() > 0:
                            right_tables_filtered.append(
                                DataFrame.from_table(
                                    right_ctx_table,
                                    base_col_names,
                                    base_dtypes,
                                    stream=s,
                                )
                            )
            if not right_tables_filtered:
                right_ctx_df = None
            elif len(right_tables_filtered) == 1:
                right_ctx_df = right_tables_filtered[0]
            else:
                with ir_context.stream_ordered_after(
                    chunk_df, *right_tables_filtered
                ) as s:
                    right_ctx_df = DataFrame.from_table(
                        plc.concatenate.concatenate(
                            [df.table for df in right_tables_filtered], stream=s
                        ),
                        base_col_names,
                        base_dtypes,
                        stream=s,
                    )
    else:
        right_ctx_df = right_halo_df if is_last_chunk else None
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
        _BOOL8,
        stream=stream,
    )
    lt = plc.binaryop.binary_operation(
        row_id,
        plc.Scalar.from_py(center_end, plc.types.SIZE_TYPE, stream=stream),
        plc.binaryop.BinaryOperator.LESS,
        _BOOL8,
        stream=stream,
    )
    mask_obj = plc.binaryop.binary_operation(
        ge, lt, plc.binaryop.BinaryOperator.LOGICAL_AND, _BOOL8, stream=stream
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
    index_dtype: plc.DataType,
    base_col_names: list[str],
    base_dtypes: list[Any],
) -> None:
    """
    Expand rolling context; each output uses paired metadata + data messages.

    Metadata carries the center row range in the combined table; the data
    :class:`Message` keeps the upstream ``sequence_number`` for the center chunk.
    """
    br = context.br()
    is_int_index = index_dtype.id() == plc.TypeId.INT64
    while True:
        if state.prefetch:
            cur_meta, cur_msg = state.prefetch.popleft()
        else:
            if (cur_meta_msg := await ch_in.recv_metadata(context)) is None:
                break
            cur_meta = ArbitraryChunk.from_message(cur_meta_msg).release()
            assert isinstance(cur_meta, _RollingStreamChunkMeta)
            cur_msg = await ch_in.recv(context)
            assert cur_msg is not None, "Expected data message after metadata."

        cur_seq = cur_msg.sequence_number

        if lookahead > 0:
            if not await _ensure_right_halo_prefetched(
                context,
                ch_in,
                state,
                cur_msg,
                br,
                is_int_index=is_int_index,
                index_col_idx=index_col_idx,
                index_dtype=index_dtype,
                lookahead=lookahead,
            ):
                continue
        else:
            if _peek_n_rows_from_message(cur_msg, br) == 0:
                continue
            await _ensure_successor_prefetched(context, ch_in, state)

        is_last_chunk = not state.prefetch
        next_msgs: Sequence[Message] | None = (
            tuple(_message_peek_copy(m, br) for _, m in state.prefetch)
            if lookahead > 0 and not is_last_chunk
            else None
        )

        prep = _prepare_expanded_rolling_frame(
            state,
            context=context,
            cur_msg=cur_msg,
            next_msgs=next_msgs,
            is_last_chunk=is_last_chunk,
            ir_context=ir_context,
            index_col_idx=index_col_idx,
            index_dtype=index_dtype,
            lookback=lookback,
            lookahead=lookahead,
            base_col_names=base_col_names,
            base_dtypes=base_dtypes,
            is_local_rank_chunk=cur_meta.is_local_rank_chunk,
        )
        assert prep is not None, (
            "empty center must be skipped before _prepare_expanded_rolling_frame"
        )
        combined_df, n_left, n_chunk_rows = prep
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

        if is_last_chunk:
            break

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
    index_col_idx: int,
    base_col_names: list[str],
    base_dtypes: list[Any],
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
        _check_ungrouped_int_chunk_order(
            state, combined, expanded, ir, index_col_idx=index_col_idx, seq=seq_num
        )
        mask = _build_center_row_mask_from_range(
            combined.num_rows,
            expanded.center_begin,
            expanded.center_end,
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

        lookback, lookahead = rolling_stream_halo_extents(
            ir.index_dtype, ir.preceding_ordinal, ir.following_ordinal
        )
        index_col_idx = base_col_names.index(ir.index.name)

        act_state = _RollingActState()
        append_done: asyncio.Queue[None] = asyncio.Queue()

        # TODO: The incoming data should be ordered on grouping
        # keys and the index column. When OrderScheme is available,
        # we should check the incoming metadata to verify this.
        # Otherwise, we can do a sort here to "correct" the order.

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
                index_dtype=ir.index_dtype,
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
                index_col_idx=index_col_idx,
                base_col_names=base_col_names,
                base_dtypes=base_dtypes,
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
