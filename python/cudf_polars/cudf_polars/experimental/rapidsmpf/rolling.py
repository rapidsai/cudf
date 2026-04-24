# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Distributed rolling window actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

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
from cudf_polars.experimental.utils import _concat

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


def _scalar_binop_scalar(
    lhs: plc.Scalar,
    rhs: plc.Scalar,
    op: plc.binaryop.BinaryOperator,
    out_dtype: plc.DataType,
    stream: Stream,
) -> plc.Scalar:
    """Apply binop using single-row columns."""
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


def _sorted_minmax_scalars(
    col: plc.Column, stream: Stream
) -> tuple[plc.Scalar, plc.Scalar]:
    """First and last values in a sorted column as scalars."""
    n = col.size()
    lo_c = plc.copying.slice(col, [0, 1], stream=stream)[0]
    hi_c = plc.copying.slice(col, [n - 1, n], stream=stream)[0]
    return lo_c.to_scalar(stream=stream), hi_c.to_scalar(stream=stream)


def _sorted_minmax_py(col: plc.Column, stream: Stream) -> tuple[int, int]:
    """First and last values in a sorted column as Python ints."""
    lo_s, hi_s = _sorted_minmax_scalars(col, stream)
    mn_val = lo_s.to_py(stream=stream)
    mx_val = hi_s.to_py(stream=stream)
    assert isinstance(mn_val, int)
    assert isinstance(mx_val, int)
    return mn_val, mx_val


def _sorted_table_slice_ge(
    table: plc.Table,
    idx_col: plc.Column,
    lower: plc.Scalar,
    *,
    stream: Stream,
) -> plc.Table:
    """Rows with ``idx_col >= lower`` via ``lower_bound`` + slice (``idx_col`` sorted ascending)."""
    hay = plc.Table([idx_col])
    needles = plc.Table([plc.Column.from_scalar(lower, 1, stream=stream)])
    lb = plc.search.lower_bound(
        hay,
        needles,
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.BEFORE],
        stream=stream,
    )
    _start_py = plc.copying.get_element(lb, 0, stream=stream).to_py(stream=stream)
    assert isinstance(_start_py, int)
    start = _start_py
    n = table.num_rows()
    (out,) = plc.copying.slice(table, [min(start, n), n], stream=stream)
    return out


def _sorted_table_slice_le(
    table: plc.Table,
    idx_col: plc.Column,
    upper: plc.Scalar,
    *,
    stream: Stream,
) -> plc.Table:
    """Return rows up to and including the upper bound."""
    hay = plc.Table([idx_col])
    needles = plc.Table([plc.Column.from_scalar(upper, 1, stream=stream)])
    ub = plc.search.upper_bound(
        hay,
        needles,
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.BEFORE],
        stream=stream,
    )
    _end_py = plc.copying.get_element(ub, 0, stream=stream).to_py(stream=stream)
    assert isinstance(_end_py, int)
    end = _end_py
    (out,) = plc.copying.slice(table, [0, max(end, 0)], stream=stream)
    return out


def _check_ungrouped_int_chunk_order(
    prev_max: int | None,
    cur_df: DataFrame,
    ir: Rolling,
    *,
    index_col_idx: int,
    seq: int,
) -> int | None:
    """Raise when the ungrouped index decreases across center chunks."""
    if ir.keys or ir.index_dtype.id() != plc.TypeId.INT64:
        return prev_max
    if cur_df.num_rows == 0:
        return prev_max
    idx = _get_idx_col(cur_df.table, index_col_idx, ir.index_dtype, cur_df.stream)
    lo, hi = _sorted_minmax_py(idx, cur_df.stream)
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


@dataclass(slots=True)
class StagedItem:
    """One staged rolling row (sequence id, table, index bounds)."""

    seq: int
    chunk: TableChunk
    bounds: tuple[int, int] | tuple[plc.Scalar, plc.Scalar] | None


@dataclass(frozen=True)
class RollingExpandedChunk:
    """Expanded chunk for rolling evaluation."""

    chunk: TableChunk
    do_evaluate_zlice: tuple[int, int]


def _merge_pending_left_ghosts_into_left_ctx(
    pending_left_dfs: list[DataFrame],
    left_ctx_df: DataFrame | None,
    *,
    ir_context: IRExecutionContext,
) -> DataFrame | None:
    """Concatenate left-ghost tables (ascending ``sequence_number``) then ``left_ctx_df``."""
    parts: list[DataFrame] = [df for df in pending_left_dfs if df.num_rows > 0]
    if left_ctx_df is not None and left_ctx_df.num_rows > 0:
        parts.append(left_ctx_df)
    if not parts:
        return left_ctx_df
    if len(parts) == 1:
        return parts[0]
    return _concat(*parts, context=ir_context)


def _chunk_index_int_bounds_from_df(
    df: DataFrame,
    *,
    index_col_idx: int,
    index_dtype: plc.DataType,
) -> tuple[int, int] | None:
    if df.num_rows == 0:
        return None
    idx = _get_idx_col(df.table, index_col_idx, index_dtype, df.stream)
    return _sorted_minmax_py(idx, df.stream)


def _chunk_index_ts_bounds_from_df(
    df: DataFrame,
    *,
    index_col_idx: int,
    index_dtype: plc.DataType,
) -> tuple[plc.Scalar, plc.Scalar] | None:
    if df.num_rows == 0:
        return None
    idx = _get_idx_col(df.table, index_col_idx, index_dtype, df.stream)
    return _sorted_minmax_scalars(idx, df.stream)


def _append_left_context_row(
    left_ctx_df: DataFrame | None,
    cur_df: DataFrame,
    *,
    lookback: int,
    ir_context: IRExecutionContext,
) -> DataFrame | None:
    if lookback <= 0:
        return left_ctx_df
    if left_ctx_df is None or left_ctx_df.num_rows == 0:
        return cur_df
    return _concat(left_ctx_df, cur_df, context=ir_context)


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


def _int_right_halo_lookahead_satisfied(
    center_max: int,
    lookahead: int,
    first_right_bounds: tuple[int, int] | None,
    last_right_bounds: tuple[int, int] | None,
) -> bool:
    """True when staged right chunks cover the INT64 lookahead edge (or first chunk is past it)."""
    edge = center_max + lookahead
    first_disjoint = first_right_bounds is not None and first_right_bounds[0] > edge
    last_reaches = last_right_bounds is not None and last_right_bounds[1] >= edge
    return first_disjoint or last_reaches


def _ts_right_halo_lookahead_satisfied(
    center_max: plc.Scalar,
    lookahead: int,
    index_dtype: plc.DataType,
    stream: Any,
    first_right_bounds: tuple[plc.Scalar, plc.Scalar] | None,
    last_right_bounds: tuple[plc.Scalar, plc.Scalar] | None,
) -> bool:
    """True when staged right chunks cover the timestamp lookahead edge (or first is past it)."""
    dur_dt = plc.DataType(_TIMESTAMP_TO_DURATION[index_dtype.id()])
    edge = _scalar_binop_scalar(
        center_max,
        plc.Scalar.from_py(lookahead, dur_dt, stream=stream),
        plc.binaryop.BinaryOperator.ADD,
        index_dtype,
        stream,
    )

    def _cmp(lhs: plc.Scalar, op: plc.binaryop.BinaryOperator) -> bool:
        return bool(
            _scalar_binop_scalar(lhs, edge, op, _BOOL8, stream).to_py(stream=stream)
        )

    first_disjoint = first_right_bounds is not None and _cmp(
        first_right_bounds[0], plc.binaryop.BinaryOperator.GREATER
    )
    last_reaches = last_right_bounds is not None and _cmp(
        last_right_bounds[1], plc.binaryop.BinaryOperator.GREATER_EQUAL
    )
    return first_disjoint or last_reaches


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
        self.lower_halo_inputs: deque[StagedItem] = deque()
        self.upper_halo_inputs: deque[StagedItem] = deque()
        self.owned_inputs: deque[StagedItem] = deque()
        self.left_ctx_df: DataFrame | None = None
        self.prev_max: int | None = None

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
    ) -> None:
        """Stage a rolling input chunk in arrival order."""
        chunk = input_chunk.chunk.make_available_and_spill(
            self.context.br(), allow_overbooking=True
        )
        bounds = self._index_bounds_for_staged_chunk(chunk)
        item = StagedItem(seq=sequence_number, chunk=chunk, bounds=bounds)
        is_ghost = input_chunk.is_ghost_chunk
        if is_ghost and not self.owned_inputs:
            self.lower_halo_inputs.append(item)
        elif is_ghost:
            self.upper_halo_inputs.append(item)
        else:
            self.owned_inputs.append(item)

    async def _has_lookahead_halo(self) -> bool:
        """Return whether staged right-hand chunks satisfy the lookahead halo."""
        if self.lookahead <= 0:
            return True
        if not self.owned_inputs:
            return False
        center = self.owned_inputs[0]
        center_bounds = center.bounds
        right_items = [*list(self.owned_inputs)[1:], *self.upper_halo_inputs]
        if not right_items or center_bounds is None:
            return False
        first_b = right_items[0].bounds
        last_b = right_items[-1].bounds
        stream = center.chunk.stream
        if self.index_dtype.id() == plc.TypeId.INT64:
            ib = cast(tuple[int, int], center_bounds)
            return _int_right_halo_lookahead_satisfied(
                ib[1],
                self.lookahead,
                cast(tuple[int, int] | None, first_b),
                cast(tuple[int, int] | None, last_b),
            )
        ib_ts = cast(tuple[plc.Scalar, plc.Scalar], center_bounds)
        return _ts_right_halo_lookahead_satisfied(
            ib_ts[1],
            self.lookahead,
            self.index_dtype,
            stream,
            cast(tuple[plc.Scalar, plc.Scalar] | None, first_b),
            cast(tuple[plc.Scalar, plc.Scalar] | None, last_b),
        )

    def _make_expanded_frame(
        self,
        left_ctx_df: DataFrame | None,
        cur_df: DataFrame,
        next_dfs: Sequence[DataFrame] | None,
        *,
        is_last_chunk: bool,
    ) -> tuple[DataFrame, int]:
        """Build an expanded input frame."""
        frame_ir = self.ir.children[0]
        chunk_table = cur_df.table
        chunk_stream = cur_df.stream
        is_int_index = self.index_dtype.id() == plc.TypeId.INT64
        assert cur_df.num_rows > 0, "Empty center chunk not supported."

        # Trim halos using index bounds from ``rolling_stream_halo_extents`` (lookback /
        # lookahead). Last chunk has no right neighbors on this rank.
        need_chunk_minmax_for_left = (
            left_ctx_df is not None and left_ctx_df.num_rows > 0 and self.lookback > 0
        )
        need_chunk_minmax_for_right = self.lookahead > 0 and not is_last_chunk
        need_chunk_minmax = need_chunk_minmax_for_left or need_chunk_minmax_for_right
        chunk_mn, chunk_mx = 0, 0
        chunk_mn_s: plc.Scalar | None = None
        chunk_mx_s: plc.Scalar | None = None
        dur_dt: plc.DataType | None = None
        base_col_names = list(frame_ir.schema.keys())
        base_dtypes = list(frame_ir.schema.values())
        if need_chunk_minmax:
            chunk_idx = _get_idx_col(
                chunk_table,
                self.index_col_idx,
                self.index_dtype,
                chunk_stream,
            )
            if is_int_index:
                chunk_mn, chunk_mx = _sorted_minmax_py(chunk_idx, chunk_stream)
            else:
                dur_dt = plc.DataType(_TIMESTAMP_TO_DURATION[self.index_dtype.id()])
                chunk_mn_s, chunk_mx_s = _sorted_minmax_scalars(chunk_idx, chunk_stream)

        if left_ctx_df is not None and left_ctx_df.num_rows > 0 and self.lookback > 0:
            left_idx = _get_idx_col(
                left_ctx_df.table,
                self.index_col_idx,
                self.index_dtype,
                left_ctx_df.stream,
            )
            if is_int_index:
                lower_s = plc.Scalar.from_py(
                    chunk_mn - self.lookback,
                    left_idx.type(),
                    stream=left_ctx_df.stream,
                )
                left_ctx_df = DataFrame.from_table(
                    _sorted_table_slice_ge(
                        left_ctx_df.table,
                        left_idx,
                        lower_s,
                        stream=left_ctx_df.stream,
                    ),
                    base_col_names,
                    base_dtypes,
                    stream=left_ctx_df.stream,
                )
            else:
                assert chunk_mn_s is not None
                assert dur_dt is not None
                lookback_s = plc.Scalar.from_py(
                    self.lookback, dur_dt, stream=chunk_stream
                )
                lower_s = _scalar_binop_scalar(
                    chunk_mn_s,
                    lookback_s,
                    plc.binaryop.BinaryOperator.SUB,
                    self.index_dtype,
                    chunk_stream,
                )
                with self.ir_context.stream_ordered_after(left_ctx_df, cur_df) as s:
                    left_ctx_df = DataFrame.from_table(
                        _sorted_table_slice_ge(
                            left_ctx_df.table,
                            left_idx,
                            lower_s,
                            stream=s,
                        ),
                        base_col_names,
                        base_dtypes,
                        stream=s,
                    )

        if self.lookahead > 0:
            if is_last_chunk:
                right_ctx_df = None
            else:
                assert next_dfs is not None
                right_tables_filtered: list[DataFrame] = []
                right_upper_int: int | None = None
                right_upper_s: plc.Scalar | None = None
                if is_int_index:
                    right_upper_int = chunk_mx + self.lookahead
                else:
                    assert chunk_mx_s is not None
                    assert dur_dt is not None
                    right_upper_s = _scalar_binop_scalar(
                        chunk_mx_s,
                        plc.Scalar.from_py(self.lookahead, dur_dt, stream=chunk_stream),
                        plc.binaryop.BinaryOperator.ADD,
                        self.index_dtype,
                        chunk_stream,
                    )
                for next_df in next_dfs:
                    next_table = next_df.table
                    next_stream = next_df.stream
                    next_idx = _get_idx_col(
                        next_table,
                        self.index_col_idx,
                        self.index_dtype,
                        next_stream,
                    )
                    if is_int_index:
                        assert right_upper_int is not None
                        upper_s = plc.Scalar.from_py(
                            right_upper_int, next_idx.type(), stream=next_stream
                        )
                        right_ctx_table = _sorted_table_slice_le(
                            next_table,
                            next_idx,
                            upper_s,
                            stream=next_stream,
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
                        assert right_upper_s is not None
                        with self.ir_context.stream_ordered_after(next_df, cur_df) as s:
                            right_ctx_table = _sorted_table_slice_le(
                                next_table,
                                next_idx,
                                right_upper_s,
                                stream=s,
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
                    with self.ir_context.stream_ordered_after(
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
        center_offset = left_ctx_df.num_rows if left_ctx_df is not None else 0
        dfs = [df for df in (left_ctx_df, cur_df, right_ctx_df) if df is not None]
        combined_df = (
            dfs[0] if len(dfs) == 1 else _concat(*dfs, context=self.ir_context)
        )

        self.left_ctx_df = _append_left_context_row(
            left_ctx_df,
            cur_df,
            lookback=self.lookback,
            ir_context=self.ir_context,
        )
        return combined_df, center_offset

    def _evict_staging_after_emit_for(
        self, *, evict_lower: bool = True, evict_upper: bool = False
    ) -> None:
        """Evict staged chunks that are no longer needed."""
        if evict_lower:
            self.lower_halo_inputs.clear()
        assert self.owned_inputs
        self.owned_inputs.popleft()
        if evict_upper:
            self.upper_halo_inputs.clear()

    def _compose_expanded_input(
        self, *, receiving: bool
    ) -> tuple[TableChunk | None, tuple[int, int] | None]:
        """Compose an expanded input chunk and slice specification."""
        frame_ir = self.ir.children[0]
        br = self.context.br()
        assert self.owned_inputs
        center = self.owned_inputs[0]
        s_curr = center.seq
        left_entries = list(self.lower_halo_inputs)
        left_dfs = [chunk_to_frame(e.chunk, frame_ir) for e in left_entries]
        left_for_prepare = _merge_pending_left_ghosts_into_left_ctx(
            left_dfs,
            self.left_ctx_df,
            ir_context=self.ir_context,
        )
        center_tbl = center.chunk
        center_df = chunk_to_frame(center_tbl, frame_ir)
        right_entries = [*list(self.owned_inputs)[1:], *self.upper_halo_inputs]

        is_last_chunk = not receiving and not right_entries
        next_dfs = (
            tuple(chunk_to_frame(e.chunk, frame_ir) for e in right_entries)
            if self.lookahead > 0 and not is_last_chunk
            else None
        )

        if (n_chunk_rows := center_df.num_rows) == 0:
            # Skip over this empty center chunk.
            self._evict_staging_after_emit_for(evict_lower=False)
            return None, None

        combined_df, center_offset = self._make_expanded_frame(
            left_for_prepare,
            center_df,
            next_dfs,
            is_last_chunk=is_last_chunk,
        )

        out_chunk = TableChunk.from_pylibcudf_table(
            combined_df.table,
            combined_df.stream,
            exclusive_view=True,
            br=br,
        )
        self.prev_max = _check_ungrouped_int_chunk_order(
            self.prev_max,
            center_df,
            self.ir,
            index_col_idx=self.index_col_idx,
            seq=s_curr,
        )
        self._evict_staging_after_emit_for(evict_upper=not receiving)
        return out_chunk, (center_offset, n_chunk_rows)

    async def prepare_output(
        self,
        *,
        receiving: bool,
    ) -> tuple[TableChunk | None, tuple[int, int] | None, int | None]:
        """Prepare an expanded output chunk and slice specification."""
        if not self.owned_inputs:
            return None, None, None

        if receiving and not await self._has_lookahead_halo():
            return None, None, None  # Need more lookahead

        seq = self.owned_inputs[0].seq
        chunk, zlice = self._compose_expanded_input(receiving=receiving)
        return chunk, zlice, seq


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
        context=context,
        ir=ir,
        ir_context=ir_context,
        lookback=lookback,
        lookahead=lookahead,
        index_col_idx=index_col_idx,
        index_dtype=index_dtype,
    )

    receiving: bool = True
    while True:
        if receiving and (msg := await ch_in.recv(context)) is not None:
            input_chunk = ArbitraryChunk.from_message(msg).release()
            assert isinstance(input_chunk, RollingInputChunk)
            await expander.add_input_chunk(input_chunk, msg.sequence_number)
        else:
            receiving = False

        chunk, zlice, seq = await expander.prepare_output(receiving=receiving)
        if chunk is not None:
            assert zlice is not None
            await ch_to_eval.send(
                context,
                Message(
                    seq,
                    ArbitraryChunk(
                        RollingExpandedChunk(
                            chunk=chunk,
                            do_evaluate_zlice=zlice,
                        )
                    ),
                ),
            )

        if not receiving and chunk is None:
            break

    await ch_to_eval.drain(context)


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
