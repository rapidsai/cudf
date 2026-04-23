# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Distributed rolling window actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Rolling
from cudf_polars.dsl.utils.windows import rolling_stream_halo_extents
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    chunk_to_frame,
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

    chunk: TableChunk
    is_ghost_chunk: bool


@dataclass(frozen=True)
class ChunkID:
    """Key for rolling input chunk."""

    sequence_number: int
    is_ghost_chunk: bool


@dataclass(frozen=True)
class RollingExpandedChunk:
    """Expanded chunk for rolling evaluation."""

    chunk: TableChunk
    do_evaluate_zlice: tuple[int, int]


def _make_expanded_chunk(
    *,
    left_ctx_df: DataFrame | None,
    center_meta: RollingInputChunk,
    center_df: DataFrame,
    next_dfs: Sequence[DataFrame] | None,
    is_last_chunk: bool,
    ir_context: IRExecutionContext,
    index_col_idx: int,
    index_dtype: plc.DataType,
    lookback: int,
    lookahead: int,
    frame_ir: IR,
    br: Any,
) -> tuple[TableChunk, tuple[int, int], DataFrame | None] | None:
    """
    Concatenate left halo, center, and right halo then pack a ``TableChunk``.

    Parameters
    ----------
    left_ctx_df
        Rows retained from earlier partitions for the left halo.
    center_meta
        Wrapper for the current center partition.
    center_df
        Decoded center table.
    next_dfs
        Decoded partitions strictly after the center used for the right halo,
        or ``None`` when this rank has no trailing partitions.
    is_last_chunk
        True when no further partitions will arrive on this rank.
    ir_context
        Execution context for CUDA stream ordering between frames.
    index_col_idx
        Rolling index column index inside each frame.
    index_dtype
        Physical dtype of the rolling index column.
    lookback
        Left halo width in host window units.
    lookahead
        Right halo width in host window units.
    frame_ir
        Child IR node whose schema matches every staged frame.
    br
        Buffer resource used when materializing the packed chunk.

    Returns
    -------
    tuple[TableChunk, tuple[int, int], DataFrame | None] | None
        ``(packed_chunk, do_evaluate_zlice, new_left_ctx)`` when the center has
        rows, otherwise ``None`` if the center is empty and must be skipped.
    """
    base_col_names = list(frame_ir.schema.keys())
    base_dtypes = list(frame_ir.schema.values())
    prep = _prepare_expanded_rolling_frame(
        left_ctx_df,
        cur_df=center_df,
        next_dfs=next_dfs,
        is_last_chunk=is_last_chunk,
        ir_context=ir_context,
        index_col_idx=index_col_idx,
        index_dtype=index_dtype,
        lookback=lookback,
        lookahead=lookahead,
        base_col_names=base_col_names,
        base_dtypes=base_dtypes,
        is_ghost_chunk=center_meta.is_ghost_chunk,
    )
    if prep is None:
        return None
    combined_df, n_left, n_chunk_rows, left_ctx_next = prep
    out_chunk = TableChunk.from_pylibcudf_table(
        combined_df.table,
        combined_df.stream,
        exclusive_view=True,
        br=br,
    )
    return (out_chunk, (n_left, n_chunk_rows), left_ctx_next)


def _merge_pending_left_ghosts_into_left_ctx(
    pending_left_dfs: list[DataFrame],
    left_ctx_df: DataFrame | None,
    *,
    ir_context: IRExecutionContext,
    frame_ir: IR,
) -> DataFrame | None:
    """Concatenate left-ghost tables (ascending ``sequence_number``) then ``left_ctx_df``."""
    base_col_names = list(frame_ir.schema.keys())
    base_dtypes = list(frame_ir.schema.values())
    parts: list[DataFrame] = [df for df in pending_left_dfs if df.num_rows > 0]
    if left_ctx_df is not None and left_ctx_df.num_rows > 0:
        parts.append(left_ctx_df)
    if not parts:
        return left_ctx_df
    if len(parts) == 1:
        return parts[0]
    with ir_context.stream_ordered_after(*parts) as s:
        return DataFrame.from_table(
            plc.concatenate.concatenate([df.table for df in parts], stream=s),
            base_col_names,
            base_dtypes,
            stream=s,
        )


# ``(sequence_number, is_ghost_chunk)`` → ``(RollingInputChunk, decoded DataFrame)``
RollingStageMap = dict[tuple[int, bool], tuple[RollingInputChunk, DataFrame]]


def _next_owned_seq(staging: RollingStageMap) -> int | None:
    owned = [k[0] for k in staging if not k[1]]
    return min(owned) if owned else None


def _left_ghost_dfs_for_center(
    staging: RollingStageMap, center_seq: int
) -> list[DataFrame]:
    keys = sorted(
        (k for k in staging if k[1] and k[0] < center_seq), key=lambda t: t[0]
    )
    return [staging[k][1] for k in keys]


def _next_dfs_right_of_center(
    staging: RollingStageMap,
    center_seq: int,
    *,
    lookahead: int,
    is_last_chunk: bool,
) -> Sequence[DataFrame] | None:
    if lookahead <= 0 or is_last_chunk:
        return None
    keys = sorted((k for k in staging if k[0] > center_seq), key=lambda t: t[0])
    return tuple(staging[k][1] for k in keys)


def _purge_staging_after_emit(staging: RollingStageMap, center_seq: int) -> None:
    for k in list(staging.keys()):
        if k[1] and k[0] < center_seq:
            del staging[k]
    staging.pop((center_seq, False), None)


async def _extend_staging_until_covered(
    staging: RollingStageMap,
    center_seq: int,
    recv_next: Callable[[], Awaitable[bool]],
    *,
    first_past_upper: Callable[[DataFrame], bool],
    last_reaches_upper: Callable[[DataFrame], bool],
) -> None:
    """Pull via ``recv_next`` into ``staging`` until the right halo is covered or EOS."""

    def _first_right_df() -> DataFrame | None:
        ks = sorted((k for k in staging if k[0] > center_seq), key=lambda t: t[0])
        return staging[ks[0]][1] if ks else None

    def _last_right_df() -> DataFrame | None:
        ks = sorted((k for k in staging if k[0] > center_seq), key=lambda t: t[0])
        return staging[ks[-1]][1] if ks else None

    if _first_right_df() is None and not await recv_next():
        return
    first = _first_right_df()
    assert first is not None
    if first_past_upper(first):
        return
    while True:
        last = _last_right_df()
        assert last is not None
        if last_reaches_upper(last):
            return
        if not await recv_next():
            return


async def _extend_staging_for_int_lookahead(
    staging: RollingStageMap,
    recv_next: Callable[[], Awaitable[bool]],
    *,
    center_seq: int,
    chunk_mx: int,
    lookahead: int,
    index_col_idx: int,
    index_dtype: plc.DataType,
) -> None:
    threshold = chunk_mx + lookahead

    def _bounds(df: DataFrame) -> tuple[int, int] | None:
        return _chunk_index_int_bounds_from_df(
            df, index_col_idx=index_col_idx, index_dtype=index_dtype
        )

    await _extend_staging_until_covered(
        staging,
        center_seq,
        recv_next,
        first_past_upper=lambda df: (b := _bounds(df)) is not None and b[0] > threshold,
        last_reaches_upper=lambda df: (b := _bounds(df)) is not None
        and b[1] >= threshold,
    )


async def _extend_staging_for_ts_lookahead(
    staging: RollingStageMap,
    recv_next: Callable[[], Awaitable[bool]],
    *,
    center_seq: int,
    chunk_mx_s: plc.Scalar,
    lookahead: int,
    index_dtype: plc.DataType,
    index_col_idx: int,
    cur_stream: Any,
) -> None:
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

    await _extend_staging_until_covered(
        staging,
        center_seq,
        recv_next,
        first_past_upper=lambda df: (b := _bounds(df)) is not None
        and _cmp(b[0], plc.binaryop.BinaryOperator.GREATER),
        last_reaches_upper=lambda df: (b := _bounds(df)) is not None
        and _cmp(b[1], plc.binaryop.BinaryOperator.GREATER_EQUAL),
    )


async def _ensure_right_halo_staged(
    staging: RollingStageMap,
    center_seq: int,
    cur_df: DataFrame,
    recv_next: Callable[[], Awaitable[bool]],
    *,
    is_int_index: bool,
    index_col_idx: int,
    index_dtype: plc.DataType,
    lookahead: int,
) -> bool:
    """Pull via ``recv_next`` until the right halo is covered or EOS."""
    if is_int_index:
        cur_bounds = _chunk_index_int_bounds_from_df(
            cur_df, index_col_idx=index_col_idx, index_dtype=index_dtype
        )
        if cur_bounds is None:
            return False
        await _extend_staging_for_int_lookahead(
            staging,
            recv_next,
            center_seq=center_seq,
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
        await _extend_staging_for_ts_lookahead(
            staging,
            recv_next,
            center_seq=center_seq,
            chunk_mx_s=chunk_mx_s,
            lookahead=lookahead,
            index_dtype=index_dtype,
            index_col_idx=index_col_idx,
            cur_stream=cur_df.stream,
        )
    return True


class _RollingChunkExpander:
    """Stage ``(sequence_number, is_ghost_chunk)`` → frame, recv until ready, emit expanded."""

    __slots__ = (
        "_staging",
        "frame_ir",
        "index_col_idx",
        "index_dtype",
        "ir",
        "left_ctx_df",
        "lookahead",
        "lookback",
        "prev_max",
        "stream_done",
    )

    def __init__(
        self,
        *,
        ir: Rolling,
        frame_ir: IR,
        lookback: int,
        lookahead: int,
        index_col_idx: int,
        index_dtype: plc.DataType,
    ) -> None:
        self.ir = ir
        self.frame_ir = frame_ir
        self.lookback = lookback
        self.lookahead = lookahead
        self.index_col_idx = index_col_idx
        self.index_dtype = index_dtype
        self._staging: RollingStageMap = {}
        self.stream_done = False
        self.left_ctx_df: DataFrame | None = None
        self.prev_max: int | None = None

    async def run(
        self,
        context: Context,
        ch_to_eval: Channel[TableChunk],
        ir_context: IRExecutionContext,
        recv_next: Callable[[], Awaitable[bool]],
    ) -> None:
        br = context.br()
        is_int_index = self.index_dtype.id() == plc.TypeId.INT64

        while True:
            while _next_owned_seq(self._staging) is None:
                if not await recv_next():
                    break
            if _next_owned_seq(self._staging) is None:
                break

            cur_seq = _next_owned_seq(self._staging)
            assert cur_seq is not None
            cur_meta, cur_df = self._staging[(cur_seq, False)]
            left_for_prepare = _merge_pending_left_ghosts_into_left_ctx(
                _left_ghost_dfs_for_center(self._staging, cur_seq),
                self.left_ctx_df,
                ir_context=ir_context,
                frame_ir=self.frame_ir,
            )

            if self.lookahead > 0:
                if not await _ensure_right_halo_staged(
                    self._staging,
                    cur_seq,
                    cur_df,
                    recv_next,
                    is_int_index=is_int_index,
                    index_col_idx=self.index_col_idx,
                    index_dtype=self.index_dtype,
                    lookahead=self.lookahead,
                ):
                    continue
            elif cur_df.num_rows == 0:
                continue
            elif not any(k[0] > cur_seq for k in self._staging):
                await recv_next()

            is_last_chunk = self.stream_done and not any(
                k[0] > cur_seq for k in self._staging
            )
            next_dfs = _next_dfs_right_of_center(
                self._staging,
                cur_seq,
                lookahead=self.lookahead,
                is_last_chunk=is_last_chunk,
            )

            built = _make_expanded_chunk(
                left_ctx_df=left_for_prepare,
                center_meta=cur_meta,
                center_df=cur_df,
                next_dfs=next_dfs,
                is_last_chunk=is_last_chunk,
                ir_context=ir_context,
                index_col_idx=self.index_col_idx,
                index_dtype=self.index_dtype,
                lookback=self.lookback,
                lookahead=self.lookahead,
                frame_ir=self.frame_ir,
                br=br,
            )
            if built is None:
                continue
            out_chunk, do_evaluate_zlice, self.left_ctx_df = built

            self.prev_max = _check_ungrouped_int_chunk_order(
                self.prev_max,
                cur_df,
                self.ir,
                index_col_idx=self.index_col_idx,
                seq=cur_seq,
            )
            await ch_to_eval.send(
                context,
                Message(
                    cur_seq,
                    ArbitraryChunk(
                        RollingExpandedChunk(
                            chunk=out_chunk,
                            do_evaluate_zlice=do_evaluate_zlice,
                        )
                    ),
                ),
            )

            _purge_staging_after_emit(self._staging, cur_seq)

            if is_last_chunk:
                break


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


class RollingChunkExpander:
    """Expander for rolling chunks."""

    def __init__(
        self,
        context: Context,
        ir: Rolling,
        ir_context: IRExecutionContext,
        lookback: int,
        lookahead: int,
        index_col_idx: int,
        index_dtype: plc.DataType,
    ):
        self.context = context
        self.ir = ir
        self.ir_context = ir_context
        self.lookback = lookback
        self.lookahead = lookahead
        self.index_col_idx = index_col_idx
        self.index_dtype = index_dtype
        self.staged_inputs: dict[ChunkID, TableChunk] = {}
        self.staged_bounds: dict[
            ChunkID, tuple[int, int] | tuple[plc.Scalar, plc.Scalar] | None
        ] = {}

    def _index_bounds_for_staged_chunk(
        self, chunk: TableChunk
    ) -> tuple[int, int] | tuple[plc.Scalar, plc.Scalar] | None:
        df = chunk_to_frame(chunk, self.ir.children[0])
        if self.index_dtype.id() == plc.TypeId.INT64:
            return _chunk_index_int_bounds_from_df(
                df,
                index_col_idx=self.index_col_idx,
                index_dtype=self.index_dtype,
            )
        return _chunk_index_ts_bounds_from_df(
            df,
            index_col_idx=self.index_col_idx,
            index_dtype=self.index_dtype,
        )

    async def add_input_chunk(
        self, input_chunk: RollingInputChunk, sequence_number: int
    ):
        key = ChunkID(sequence_number, input_chunk.is_ghost_chunk)
        chunk = input_chunk.chunk.make_available_and_spill(
            self.context.br(), allow_overbooking=True
        )
        self.staged_inputs[key] = chunk
        self.staged_bounds[key] = self._index_bounds_for_staged_chunk(chunk)

    def _get_next_chunk_id(self) -> ChunkID | None:
        """Get the next owned chunk ID."""
        owned = [k for k in self.staged_inputs if not k.is_ghost_chunk]
        if not owned:
            return None
        seq = min(k.sequence_number for k in owned)
        return ChunkID(seq, False)

    async def _has_lookahead_halo(self, chunk_id: ChunkID) -> bool:
        """Check if we have enough lookahead chunks."""
        if self.lookahead <= 0:
            return True
        keys = sorted(
            (
                k
                for k in self.staged_inputs
                if k.sequence_number > chunk_id.sequence_number
            ),
            key=lambda k: k.sequence_number,
        )
        if not keys or (cb := self.staged_bounds.get(chunk_id)) is None:
            return False
        fb, lb = self.staged_bounds[keys[0]], self.staged_bounds[keys[-1]]
        if self.index_dtype.id() == plc.TypeId.INT64:
            th = cb[1] + self.lookahead
            return (fb is not None and fb[0] > th) or (lb is not None and lb[1] >= th)
        st = self.staged_inputs[chunk_id].stream
        dur = _duration_dtype_for_timestamp(self.index_dtype)
        upper = _scalar_binop_scalar(
            cb[1],
            plc.Scalar.from_py(self.lookahead, dur, stream=st),
            plc.binaryop.BinaryOperator.ADD,
            self.index_dtype,
            st,
        )
        return (
            fb is not None
            and bool(
                _scalar_binop_scalar(
                    fb[0], upper, plc.binaryop.BinaryOperator.GREATER, _BOOL8, st
                ).to_py(stream=st)
            )
        ) or (
            lb is not None
            and bool(
                _scalar_binop_scalar(
                    lb[1], upper, plc.binaryop.BinaryOperator.GREATER_EQUAL, _BOOL8, st
                ).to_py(stream=st)
            )
        )

    async def prepare_output(
        self,
        receiving: bool,
    ) -> tuple[TableChunk | None, tuple[int, int] | None, int | None]:
        """Prepare an expanded output chunk and slice specification."""
        if (chunk_id := self._get_next_chunk_id()) is None:
            # No "owned" chunks available yet.
            return None, None, None

        # We have an "owned" chunk available.
        # Check if we also have enough lookahead chunks.
        if (
            receiving
            and self.lookahead > 0
            and not await self._has_lookahead_halo(chunk_id)
        ):
            # Need more lookahead chunks to progress.
            return None, None, None

        # Pop or copy/gather rows from self.staged_inputs
        # to construct the expanded chunk for chunk_id.
        # We need to extract the slice specification for
        # the center of the expanded chunk.
        chunk, zlice = self._make_expanded_chunk(chunk_id)

        return chunk, zlice, chunk_id.sequence_number


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
) -> None:
    """Expand chunks with halos and send them for rolling evaluation."""
    expander = RollingChunkExpander(
        ir=ir,
        ir_context=ir_context,
        lookback=lookback,
        lookahead=lookahead,
        index_col_idx=index_col_idx,
        index_dtype=index_dtype,
    )

    receiving: bool = True
    chunk_id: int | None = None
    while True:
        if receiving and (msg := await ch_in.recv(context)) is not None:
            input_chunk = ArbitraryChunk.from_message(msg).release()
            assert isinstance(input_chunk, RollingInputChunk)
            await expander.add_input_chunk(input_chunk, msg.sequence_number)
        else:
            receiving = False

        chunk, chunk_id, zlice = await expander.prepare_output(receiving)
        if chunk is not None:
            assert chunk_id is not None
            await ch_to_eval.send(
                context,
                Message(
                    chunk_id,
                    ArbitraryChunk(
                        RollingExpandedChunk(
                            chunk=chunk,
                            do_evaluate_zlice=zlice,
                        )
                    ),
                ),
            )

        if not receiving:
            break

    await ch_to_eval.drain(context)

    # expander = _RollingChunkExpander(
    #     ir=ir,
    #     frame_ir=frame_ir,
    #     lookback=lookback,
    #     lookahead=lookahead,
    #     index_col_idx=index_col_idx,
    #     index_dtype=index_dtype,
    # )
    # br = context.br()

    # async def recv_next_into_staging() -> bool:
    #     msg = await ch_in.recv(context)
    #     if msg is None:
    #         expander.stream_done = True
    #         return False
    #     seq = msg.sequence_number
    #     inner = ArbitraryChunk.from_message(msg).release()
    #     assert isinstance(inner, RollingInputChunk)
    #     chunk = inner.chunk.make_available_and_spill(br, allow_overbooking=True)
    #     df = chunk_to_frame(chunk, frame_ir)
    #     expander._staging[(seq, inner.is_ghost_chunk)] = (inner, df)
    #     return True

    # await expander.run(context, ch_to_eval, ir_context, recv_next_into_staging)
    # await ch_to_eval.drain(context)


async def eval_and_send(
    context: Context,
    ch_from_extend: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    ir: Rolling,
    ir_context: IRExecutionContext,
    *,
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
        result_df = await asyncio.to_thread(
            Rolling.do_evaluate,
            *_non_child_args_static,
            inner.do_evaluate_zlice,
            chunk_to_frame(chunk, ir.children[0]),
            context=ir_context,
        )
        result_chunk = TableChunk.from_pylibcudf_table(
            result_df.table,
            result_df.stream,
            exclusive_view=True,
            br=context.br(),
        )
        if tracer is not None:
            tracer.add_chunk(table=result_chunk.table_view())
        await ch_out.send(context, Message(seq, result_chunk))

    await ch_out.drain(context)


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

        frame_ir = ir.children[0]
        lookback, lookahead = rolling_stream_halo_extents(
            ir.index_dtype, ir.preceding_ordinal, ir.following_ordinal
        )
        index_col_idx = list(frame_ir.schema.keys()).index(ir.index.name)

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
                frame_ir=frame_ir,
            ),
            eval_and_send(
                context,
                ch_to_eval,
                ch_out,
                ir,
                ir_context,
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
