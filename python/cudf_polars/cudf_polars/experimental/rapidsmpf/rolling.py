# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Distributed rolling window actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

from cudf_polars.containers import DataFrame
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
    from collections.abc import Callable, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer


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
    """Apply ``op`` to two scalars using one-row columns. Output dtype is ``out_dtype``."""
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


def _minmax_scalars(col: plc.Column, stream: Stream) -> tuple[plc.Scalar, plc.Scalar]:
    """
    First and last values in ``col`` as scalars.

    Index must be sorted in ascending order.
    """
    n = col.size()
    lo_c = plc.copying.slice(col, [0, 1], stream=stream)[0]
    hi_c = plc.copying.slice(col, [n - 1, n], stream=stream)[0]
    return lo_c.to_scalar(stream=stream), hi_c.to_scalar(stream=stream)


def _py_minmax_sorted(col: plc.Column, stream: Stream) -> tuple[int, int]:
    """
    First and last values in ``col`` as Python ints.

    Expects INT64 index sorted in ascending order.
    """
    lo_s, hi_s = _minmax_scalars(col, stream)
    mn_val = lo_s.to_py(stream=stream)
    mx_val = hi_s.to_py(stream=stream)
    assert isinstance(mn_val, int)
    assert isinstance(mx_val, int)
    return mn_val, mx_val


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


def _check_ungrouped_int_chunk_order(
    prev_max: int | None,
    cur_df: DataFrame,
    ir: Rolling,
    *,
    index_col_idx: int,
    seq: int,
) -> int | None:
    """Raise when the ungrouped INT64 index decreases across center chunks."""
    if ir.keys or ir.index_dtype.id() != plc.TypeId.INT64:
        return prev_max
    if cur_df.num_rows == 0:
        return prev_max
    idx = _get_idx_col(cur_df.table, index_col_idx, ir.index_dtype, cur_df.stream)
    lo, hi = _py_minmax_sorted(idx, cur_df.stream)
    if prev_max is not None and lo < prev_max:
        raise RuntimeError(
            f"rolling streaming: INT64 index decreased across chunks "
            f"(min {lo} < prev max {prev_max}, seq={seq})"
        )  # pragma: no cover; Should never get here
    return hi


@dataclass(frozen=True)
class RollingInputChunk:
    """Input chunk for rolling evaluation."""

    is_ghost_chunk: bool
    chunk: TableChunk


@dataclass(frozen=True)
class RollingExpandedChunk:
    """Expanded chunk for rolling evaluation."""

    chunk: TableChunk
    do_evaluate_zlice: tuple[int, int]


def _fused_expanded_zlice(
    center_begin: int,
    n_center: int,
    n_center_prior: int,
    global_zlice: tuple[int, int | None] | None,
) -> tuple[int, int]:
    """Row offset and length in expanded-frame coordinates for ``Rolling.do_evaluate``."""
    if global_zlice is None:
        return (center_begin, n_center)
    off, length = global_zlice
    lo = max(0, off - n_center_prior)
    hi = n_center if length is None else min(n_center, off + length - n_center_prior)
    if lo >= hi:
        return (center_begin, 0)
    return (center_begin + lo, hi - lo)


async def _recv_rolling_input(
    context: Context,
    ch_in: Channel[TableChunk],
    br: Any,
    col_names: list[str],
    col_dtypes: list[Any],
) -> tuple[RollingInputChunk, int, DataFrame] | None:
    msg = await ch_in.recv(context)
    if msg is None:
        return None
    seq = msg.sequence_number
    inner = ArbitraryChunk.from_message(msg).release()
    assert isinstance(inner, RollingInputChunk)
    chunk = inner.chunk.make_available_and_spill(br, allow_overbooking=True)
    df = DataFrame.from_table(
        chunk.table_view(), col_names, col_dtypes, stream=chunk.stream
    )
    return (inner, seq, df)


def _chunk_index_int_bounds_from_df(
    df: DataFrame,
    *,
    index_col_idx: int,
    index_dtype: plc.DataType,
) -> tuple[int, int] | None:
    if df.num_rows == 0:
        return None
    idx = _get_idx_col(df.table, index_col_idx, index_dtype, df.stream)
    return _py_minmax_sorted(idx, df.stream)


def _chunk_index_ts_bounds_from_df(
    df: DataFrame,
    *,
    index_col_idx: int,
    index_dtype: plc.DataType,
) -> tuple[plc.Scalar, plc.Scalar] | None:
    if df.num_rows == 0:
        return None
    idx = _get_idx_col(df.table, index_col_idx, index_dtype, df.stream)
    return _minmax_scalars(idx, df.stream)


async def _extend_prefetch_until_covered(
    context: Context,
    ch_in: Channel[TableChunk],
    prefetch: deque[tuple[RollingInputChunk, int, DataFrame]],
    br: Any,
    col_names: list[str],
    col_dtypes: list[Any],
    *,
    first_past_upper: Callable[[DataFrame], bool],
    last_reaches_upper: Callable[[DataFrame], bool],
) -> None:
    """Grow prefetch until the right halo is covered or the input stream ends."""
    if not prefetch and not await _recv_one_into_prefetch(
        context, ch_in, prefetch, br, col_names, col_dtypes
    ):
        return
    if first_past_upper(prefetch[0][2]):
        return
    while True:
        if last_reaches_upper(prefetch[-1][2]):
            return
        if not await _recv_one_into_prefetch(
            context, ch_in, prefetch, br, col_names, col_dtypes
        ):
            return


async def _recv_one_into_prefetch(
    context: Context,
    ch_in: Channel[TableChunk],
    prefetch: deque[tuple[RollingInputChunk, int, DataFrame]],
    br: Any,
    col_names: list[str],
    col_dtypes: list[Any],
) -> bool:
    """Decode one chunk and append it to ``prefetch``. Returns ``False`` at end of stream."""
    got = await _recv_rolling_input(context, ch_in, br, col_names, col_dtypes)
    if got is None:
        return False
    prefetch.append(got)
    return True


async def _extend_prefetch_for_int_lookahead(
    context: Context,
    ch_in: Channel[TableChunk],
    prefetch: deque[tuple[RollingInputChunk, int, DataFrame]],
    br: Any,
    col_names: list[str],
    col_dtypes: list[Any],
    *,
    chunk_mx: int,
    lookahead: int,
    index_col_idx: int,
    index_dtype: plc.DataType,
) -> None:
    """Prefetch int-index chunks until the right halo is covered or the stream ends."""
    threshold = chunk_mx + lookahead

    def _bounds(df: DataFrame) -> tuple[int, int] | None:
        return _chunk_index_int_bounds_from_df(
            df, index_col_idx=index_col_idx, index_dtype=index_dtype
        )

    await _extend_prefetch_until_covered(
        context,
        ch_in,
        prefetch,
        br,
        col_names,
        col_dtypes,
        first_past_upper=lambda df: (b := _bounds(df)) is not None and b[0] > threshold,
        last_reaches_upper=lambda df: (b := _bounds(df)) is not None
        and b[1] >= threshold,
    )


async def _extend_prefetch_for_ts_lookahead(
    context: Context,
    ch_in: Channel[TableChunk],
    prefetch: deque[tuple[RollingInputChunk, int, DataFrame]],
    br: Any,
    col_names: list[str],
    col_dtypes: list[Any],
    *,
    chunk_mx_s: plc.Scalar,
    lookahead: int,
    index_dtype: plc.DataType,
    index_col_idx: int,
    cur_stream: Any,
) -> None:
    """Prefetch timestamp-index chunks until the right halo is covered or the stream ends."""
    dur_dt = _duration_dtype_for_timestamp(index_dtype)
    upper_s = _scalar_binop_scalar(
        chunk_mx_s,
        plc.Scalar.from_py(lookahead, dur_dt, stream=cur_stream),
        plc.binaryop.BinaryOperator.ADD,
        index_dtype,
        cur_stream,
    )

    def _bounds(df: DataFrame) -> tuple[plc.Scalar, plc.Scalar] | None:
        return _chunk_index_ts_bounds_from_df(
            df, index_col_idx=index_col_idx, index_dtype=index_dtype
        )

    def _cmp(lhs: plc.Scalar, op: plc.binaryop.BinaryOperator) -> bool:
        return bool(
            _scalar_binop_scalar(lhs, upper_s, op, _BOOL8, cur_stream).to_py(
                stream=cur_stream
            )
        )

    await _extend_prefetch_until_covered(
        context,
        ch_in,
        prefetch,
        br,
        col_names,
        col_dtypes,
        first_past_upper=lambda df: (b := _bounds(df)) is not None
        and _cmp(b[0], plc.binaryop.BinaryOperator.GREATER),
        last_reaches_upper=lambda df: (b := _bounds(df)) is not None
        and _cmp(b[1], plc.binaryop.BinaryOperator.GREATER_EQUAL),
    )


async def _ensure_right_halo_prefetched(
    context: Context,
    ch_in: Channel[TableChunk],
    prefetch: deque[tuple[RollingInputChunk, int, DataFrame]],
    cur_df: DataFrame,
    br: Any,
    *,
    is_int_index: bool,
    index_col_idx: int,
    index_dtype: plc.DataType,
    lookahead: int,
    col_names: list[str],
    col_dtypes: list[Any],
) -> bool:
    """Prefetch the right halo. Returns ``False`` when the center chunk is empty."""
    if is_int_index:
        cur_bounds = _chunk_index_int_bounds_from_df(
            cur_df, index_col_idx=index_col_idx, index_dtype=index_dtype
        )
        if cur_bounds is None:
            return False
        await _extend_prefetch_for_int_lookahead(
            context,
            ch_in,
            prefetch,
            br,
            col_names,
            col_dtypes,
            chunk_mx=cur_bounds[1],
            lookahead=lookahead,
            index_col_idx=index_col_idx,
            index_dtype=index_dtype,
        )
    else:
        if cur_df.num_rows == 0:
            return False
        cur_idx = _get_idx_col(cur_df.table, index_col_idx, index_dtype, cur_df.stream)
        _, chunk_mx_s = _minmax_scalars(cur_idx, cur_df.stream)
        await _extend_prefetch_for_ts_lookahead(
            context,
            ch_in,
            prefetch,
            br,
            col_names,
            col_dtypes,
            chunk_mx_s=chunk_mx_s,
            lookahead=lookahead,
            index_dtype=index_dtype,
            index_col_idx=index_col_idx,
            cur_stream=cur_df.stream,
        )
    return True


def _prepare_expanded_rolling_frame(
    left_ctx_df: DataFrame | None,
    *,
    cur_df: DataFrame,
    next_dfs: Sequence[DataFrame] | None,
    is_last_chunk: bool,
    ir_context: IRExecutionContext,
    index_col_idx: int,
    index_dtype: plc.DataType,
    lookback: int,
    lookahead: int,
    base_col_names: list[str],
    base_dtypes: list[Any],
    is_ghost_chunk: bool,
) -> tuple[DataFrame, int, int, DataFrame | None] | None:
    """Build the left, center, and right rolling frame. Updates ``left_ctx`` from ``cur_df``."""
    if is_ghost_chunk:
        raise NotImplementedError(
            "Non-local rank boundary chunks are not supported for rolling yet."
        )

    n_chunk_rows = cur_df.num_rows
    if n_chunk_rows == 0:
        return None

    chunk_table = cur_df.table
    chunk_stream = cur_df.stream
    is_int_index = index_dtype.id() == plc.TypeId.INT64

    # Trim halos using index bounds from ``rolling_stream_halo_extents`` (lookback /
    # lookahead). Last chunk has no right neighbors on this rank.
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
            chunk_mn, chunk_mx = _py_minmax_sorted(chunk_idx, chunk_stream)
        else:
            dur_dt = _duration_dtype_for_timestamp(index_dtype)
            chunk_mn_s, chunk_mx_s = _minmax_scalars(chunk_idx, chunk_stream)

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
            with ir_context.stream_ordered_after(left_ctx_df, cur_df) as s:
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
            right_ctx_df = None
        else:
            assert next_dfs is not None
            right_tables_filtered: list[DataFrame] = []
            right_upper: int | plc.Scalar
            if is_int_index:
                right_upper = chunk_mx + lookahead
            else:
                assert chunk_mx_s is not None
                assert dur_dt is not None
                right_upper = _scalar_binop_scalar(
                    chunk_mx_s,
                    plc.Scalar.from_py(lookahead, dur_dt, stream=chunk_stream),
                    plc.binaryop.BinaryOperator.ADD,
                    index_dtype,
                    chunk_stream,
                )
            for next_df in next_dfs:
                next_table = next_df.table
                next_stream = next_df.stream
                next_idx = _get_idx_col(
                    next_table, index_col_idx, index_dtype, next_stream
                )
                if is_int_index:
                    right_ctx_table = _filter_threshold(
                        next_table,
                        next_idx,
                        right_upper,
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
                    with ir_context.stream_ordered_after(next_df, cur_df) as s:
                        right_ctx_table = _filter_threshold(
                            next_table,
                            next_idx,
                            right_upper,
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
                    cur_df, *right_tables_filtered
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
        right_ctx_df = None
    n_left = left_ctx_df.num_rows if left_ctx_df is not None else 0
    dfs = [df for df in (left_ctx_df, cur_df, right_ctx_df) if df is not None]
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

    left_ctx_next = _append_left_context_row(
        left_ctx_df,
        cur_df,
        lookback=lookback,
        ir_context=ir_context,
        col_names=base_col_names,
        col_dtypes=base_dtypes,
    )
    return combined_df, n_left, n_chunk_rows, left_ctx_next


def _append_left_context_row(
    left_ctx_df: DataFrame | None,
    cur_df: DataFrame,
    *,
    lookback: int,
    ir_context: IRExecutionContext,
    col_names: list[str],
    col_dtypes: list[Any],
) -> DataFrame | None:
    if lookback <= 0:
        return left_ctx_df
    if left_ctx_df is None or left_ctx_df.num_rows == 0:
        return cur_df
    with ir_context.stream_ordered_after(left_ctx_df, cur_df) as s:
        return DataFrame.from_table(
            plc.concatenate.concatenate([left_ctx_df.table, cur_df.table], stream=s),
            col_names,
            col_dtypes,
            stream=s,
        )


async def add_ghost_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    collective_ids: list[int],
) -> None:
    """Add ghost chunks to the rolling stream."""
    assert len(collective_ids), "Placeholder for multi-rank support."
    br = context.br()
    while (msg := await ch_in.recv(context)) is not None:
        await ch_out.send(
            context,
            Message(
                msg.sequence_number,
                ArbitraryChunk(
                    RollingInputChunk(
                        chunk=TableChunk.from_message(msg, br),
                        is_ghost_chunk=False,
                    )
                ),
            ),
        )

    await ch_out.drain(context)


async def expand_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_to_eval: Channel[TableChunk],
    ir: Rolling,
    ir_context: IRExecutionContext,
    *,
    lookback: int,
    lookahead: int,
    index_col_idx: int,
    index_dtype: plc.DataType,
    base_col_names: list[str],
    base_dtypes: list[Any],
) -> None:
    """Expand chunks with halos and send them for rolling evaluation."""
    prefetch: deque[tuple[RollingInputChunk, int, DataFrame]] = deque()
    left_ctx_df: DataFrame | None = None
    prev_max: int | None = None
    n_center_prior = 0
    assert ir.zlice is None, "Expected no zlice in rolling actor."
    # Negative-offset zlice (e.g. from .tail()) cannot be resolved without knowing the
    # total row count. Pass None so all center rows are evaluated; eval_and_send
    # buffers the results and applies the tail-slice after seeing all chunks.
    zlice = ir.zlice
    if zlice is not None and zlice[0] < 0:
        zlice = None
    br = context.br()
    is_int_index = index_dtype.id() == plc.TypeId.INT64

    while True:
        if prefetch:
            cur_meta, cur_seq, cur_df = prefetch.popleft()
        else:
            got = await _recv_rolling_input(
                context, ch_in, br, base_col_names, base_dtypes
            )
            if got is None:
                break
            cur_meta, cur_seq, cur_df = got

        if lookahead > 0:
            if not await _ensure_right_halo_prefetched(
                context,
                ch_in,
                prefetch,
                cur_df,
                br,
                is_int_index=is_int_index,
                index_col_idx=index_col_idx,
                index_dtype=index_dtype,
                lookahead=lookahead,
                col_names=base_col_names,
                col_dtypes=base_dtypes,
            ):
                continue
        elif cur_df.num_rows == 0:
            continue  # no center rows to send
        elif not prefetch:
            await _recv_one_into_prefetch(
                context, ch_in, prefetch, br, base_col_names, base_dtypes
            )

        is_last_chunk = not prefetch
        next_dfs: Sequence[DataFrame] | None = (
            tuple(df for _, _, df in prefetch)
            if lookahead > 0 and not is_last_chunk
            else None
        )

        prep = _prepare_expanded_rolling_frame(
            left_ctx_df,
            cur_df=cur_df,
            next_dfs=next_dfs,
            is_last_chunk=is_last_chunk,
            ir_context=ir_context,
            index_col_idx=index_col_idx,
            index_dtype=index_dtype,
            lookback=lookback,
            lookahead=lookahead,
            base_col_names=base_col_names,
            base_dtypes=base_dtypes,
            is_ghost_chunk=cur_meta.is_ghost_chunk,
        )
        assert prep is not None, (
            "empty center must be skipped before _prepare_expanded_rolling_frame"
        )
        combined_df, n_left, n_chunk_rows, left_ctx_df = prep

        prev_max = _check_ungrouped_int_chunk_order(
            prev_max,
            cur_df,
            ir,
            index_col_idx=index_col_idx,
            seq=cur_seq,
        )
        out_chunk = TableChunk.from_pylibcudf_table(
            combined_df.table,
            combined_df.stream,
            exclusive_view=True,
            br=br,
        )
        await ch_to_eval.send(
            context,
            Message(
                cur_seq,
                ArbitraryChunk(
                    RollingExpandedChunk(
                        chunk=out_chunk,
                        do_evaluate_zlice=_fused_expanded_zlice(
                            n_left, n_chunk_rows, n_center_prior, zlice
                        ),
                    )
                ),
            ),
        )
        n_center_prior += n_chunk_rows

        if is_last_chunk:
            break

    await ch_to_eval.drain(context)


async def eval_and_send(
    context: Context,
    ch_from_extend: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    ir: Rolling,
    ir_context: IRExecutionContext,
    *,
    base_col_names: list[str],
    base_dtypes: list[Any],
    tracer: ActorTracer | None,
) -> None:
    """Receive expanded chunks and per-chunk zlice metadata, evaluate rolling, send downstream."""
    _non_child_args_static = ir._non_child_args[:-1]
    while (msg := await ch_from_extend.recv(context)) is not None:
        seq = msg.sequence_number
        inner = ArbitraryChunk.from_message(msg).release()
        assert isinstance(inner, RollingExpandedChunk)
        chunk = inner.chunk.make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        combined = DataFrame.from_table(
            chunk.table_view(),
            base_col_names,
            base_dtypes,
            stream=chunk.stream,
        )
        result_df = await asyncio.to_thread(
            Rolling.do_evaluate,
            *_non_child_args_static,
            inner.do_evaluate_zlice,
            combined,
            context=ir_context,
        )
        await _rolling_send_chunk(
            context, ch_out, result_df, sequence_number=seq, tracer=tracer
        )

    await ch_out.drain(context)


async def _rolling_send_chunk(
    context: Context,
    ch_out: Channel[TableChunk],
    result_df: DataFrame,
    *,
    sequence_number: int,
    tracer: ActorTracer | None,
) -> None:
    result_chunk = TableChunk.from_pylibcudf_table(
        result_df.table,
        result_df.stream,
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
    """Streaming rolling actor. Rank boundaries, expand halos, evaluate, and send."""
    assert comm.nranks == 1
    ch_after_boundaries: Channel[TableChunk] = context.create_channel()
    ch_to_eval: Channel[TableChunk] = context.create_channel()
    async with shutdown_on_error(
        context,
        ch_in,
        ch_out,
        ch_after_boundaries,
        ch_to_eval,
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

        # TODO: The incoming data should be ordered on grouping
        # keys and the index column. When OrderScheme is available,
        # we should check the incoming metadata to verify this.
        # Otherwise, we can do a sort here to "correct" the order.

        await gather_in_task_group(
            add_ghost_chunks(
                context,
                ch_in,
                ch_after_boundaries,
                collective_ids,
            ),
            expand_chunks(
                context,
                ch_after_boundaries,
                ch_to_eval,
                ir,
                ir_context,
                lookback=lookback,
                lookahead=lookahead,
                index_col_idx=index_col_idx,
                index_dtype=ir.index_dtype,
                base_col_names=base_col_names,
                base_dtypes=base_dtypes,
            ),
            eval_and_send(
                context,
                ch_to_eval,
                ch_out,
                ir,
                ir_context,
                base_col_names=base_col_names,
                base_dtypes=base_dtypes,
                tracer=tracer,
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
