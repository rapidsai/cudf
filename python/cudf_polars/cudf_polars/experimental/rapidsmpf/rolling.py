# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Distributed rolling window actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import Rolling
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
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


@define_actor()
async def prepare_rank_boundaries_actor(
    context: Context,
    comm: Communicator,
    ir: Rolling,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    collective_ids: list[int],
) -> None:
    """Forward sorted rolling input for single rank. Multi rank TBD (SparseAlltoall)."""
    assert comm.nranks == 1
    assert len(collective_ids) == 2
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ):
        metadata_in = await recv_metadata(ch_in, context)
        await send_metadata(ch_out, context, metadata_in)
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
        await ch_out.drain(context)


@define_actor()
async def rolling_actor(
    context: Context,
    comm: Communicator,
    ir: Rolling,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
) -> None:
    """Buffer chunks, run range rolling, strip context rows, apply zlice."""
    assert comm.nranks == 1
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        metadata_in = await recv_metadata(ch_in, context)
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(local_count=metadata_in.local_count, partitioning=None),
        )

        # Buffer all incoming chunks in spillable messages
        sm = context.spillable_messages()
        mids: list[int] = []
        while (msg := await ch_in.recv(context)) is not None:
            mids.append(sm.insert(msg))

        n_chunks = len(mids)
        col_names = list(ir.children[0].schema.keys())
        col_dtypes = list(ir.children[0].schema.values())
        br = context.br()

        if n_chunks == 0:
            await ch_out.drain(context)
            return

        type_id = ir.index_dtype.id()
        preceding_native = _ordinal_to_native(type_id, ir.preceding_ordinal)
        following_native = _ordinal_to_native(type_id, ir.following_ordinal)
        lookback = max(0, -preceding_native)
        lookahead = max(0, preceding_native + following_native)
        index_col_idx = col_names.index(ir.index.name)
        i64 = plc.DataType(plc.TypeId.INT64)
        bool8 = plc.DataType(plc.TypeId.BOOL8)

        left_ctx_df: DataFrame | None = None
        right_halo_df: DataFrame | None = None

        non_child_args_no_zlice = ir._non_child_args[:-1] + (None,)
        n_processed = 0  # running tally of input rows processed (for zlice)
        for i, mid in enumerate(mids):
            chunk = TableChunk.from_message(sm.extract(mid=mid), br)
            chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
            chunk_table = chunk.table_view()
            n_chunk_rows = chunk_table.num_rows()

            if n_chunk_rows == 0:
                # Empty chunks carry no output rows and contribute nothing to
                # n_processed, so skipping them is safe even when zlice is set.
                continue

            chunk_stream = chunk.stream

            # Compute chunk min/max once if needed for left_ctx trim or right lookahead
            need_chunk_minmax = (
                left_ctx_df is not None and left_ctx_df.num_rows > 0 and lookback > 0
            ) or (lookahead > 0 and i < n_chunks - 1)
            if need_chunk_minmax:
                chunk_idx = _get_idx_col_i64(
                    chunk_table, index_col_idx, type_id, i64, chunk_stream
                )
                chunk_mn, chunk_mx = _minmax_py(chunk_idx, i64, chunk_stream)

            # Trim left_ctx: drop rows with index < (chunk_min - lookback)
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
                    col_names,
                    col_dtypes,
                    stream=left_ctx_df.stream,
                )

            # Build right context for this chunk
            right_ctx_df: DataFrame | None
            if i == n_chunks - 1:
                right_ctx_df = right_halo_df
            elif lookahead > 0:
                next_chunk = TableChunk.from_message(sm.extract(mid=mids[i + 1]), br)
                next_chunk = next_chunk.make_available_and_spill(
                    br, allow_overbooking=True
                )
                next_table = next_chunk.table_view()
                next_stream = next_chunk.stream
                next_idx = _get_idx_col_i64(
                    next_table, index_col_idx, type_id, i64, next_stream
                )
                right_ctx_table = _filter_le(
                    next_table, next_idx, chunk_mx + lookahead, i64, bool8, next_stream
                )
                mids[i + 1] = sm.insert(Message(0, next_chunk))
                right_ctx_df = (
                    DataFrame.from_table(
                        right_ctx_table, col_names, col_dtypes, stream=next_stream
                    )
                    if right_ctx_table.num_rows() > 0
                    else None
                )
            else:
                right_ctx_df = None

            # Assemble left_ctx + chunk + right_ctx, joining streams correctly
            chunk_df = DataFrame.from_table(
                chunk_table, col_names, col_dtypes, stream=chunk_stream
            )
            n_left = left_ctx_df.num_rows if left_ctx_df is not None else 0
            dfs = [df for df in (left_ctx_df, chunk_df, right_ctx_df) if df is not None]
            if len(dfs) == 1:
                combined_df = dfs[0]
            else:
                with ir_context.stream_ordered_after(*dfs) as s:
                    combined_df = DataFrame.from_table(
                        plc.concatenate.concatenate([df.table for df in dfs], stream=s),
                        col_names,
                        col_dtypes,
                        stream=s,
                    )

            result_df = await asyncio.to_thread(
                Rolling.do_evaluate,
                *non_child_args_no_zlice,
                combined_df,
                context=ir_context,
            )

            # Strip context rows to recover only this chunk's output rows
            chunk_result_df = result_df.slice((n_left, n_chunk_rows))
            local_result_df: DataFrame | None = chunk_result_df

            # Apply zlice across the streamed output
            if ir.zlice is not None:
                zlice_offset, zlice_length = ir.zlice
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

            n_processed += n_chunk_rows

            # Update left context: append current chunk (trimmed on next iteration)
            if lookback > 0:
                if left_ctx_df is None or left_ctx_df.num_rows == 0:
                    left_ctx_df = chunk_df
                else:
                    with ir_context.stream_ordered_after(left_ctx_df, chunk_df) as s:
                        left_ctx_df = DataFrame.from_table(
                            plc.concatenate.concatenate(
                                [left_ctx_df.table, chunk_df.table], stream=s
                            ),
                            col_names,
                            col_dtypes,
                            stream=s,
                        )

            if local_result_df is not None and local_result_df.num_rows > 0:
                result_chunk = TableChunk.from_pylibcudf_table(
                    local_result_df.table,
                    local_result_df.stream,
                    exclusive_view=True,
                    br=br,
                )
                if tracer is not None:
                    tracer.add_chunk(table=result_chunk.table_view())
                await ch_out.send(context, Message(0, result_chunk))

        await ch_out.drain(context)


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
    ch_rolling_in = ctx.create_channel()
    nodes[ir] = [
        prepare_rank_boundaries_actor(
            ctx,
            comm,
            ir,
            rec.state["ir_context"],
            ch_in=channels[ir.children[0]].reserve_output_slot(),
            ch_out=ch_rolling_in,
            collective_ids=list(collective_ids),
        ),
        rolling_actor(
            ctx,
            comm,
            ir,
            rec.state["ir_context"],
            ch_in=ch_rolling_in,
            ch_out=channels[ir].reserve_input_slot(),
        ),
    ]
    return nodes, channels
