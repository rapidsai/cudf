# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Distributed rolling window actor for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.halo_exchange import HaloExchange
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc
from pylibcudf.contiguous_split import pack

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import Rolling
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    allgather_reduce,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.utils.cuda_stream import get_joined_cuda_stream

if TYPE_CHECKING:
    from collections.abc import Callable

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.spillable_messages import SpillableMessages

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


def _ordinal_to_native(type_id: plc.TypeId, ordinal: int) -> int:
    """Convert a raw polars ordinal to native column units."""
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


def _filter_relay_pd(
    pd: PackedData,
    index_col_idx: int,
    type_id: plc.TypeId,
    threshold: int,
    stream: Stream,
    br: Any,
    *,
    ge: bool,
) -> PackedData | None:
    """
    Unpack, filter, and repack a halo payload for multi-hop relay.

    Parameters
    ----------
    pd
        Packed halo payload to filter.
    index_col_idx
        Column position of the rolling index within the table.
    type_id
        TypeId of the index column (used to cast to INT64 for comparisons).
    threshold
        Index threshold.  Rows with index >= threshold are kept when
        ``ge=True``; rows with index <= threshold are kept when ``ge=False``.
    stream
        CUDA stream for all GPU operations.
    br
        Buffer resource used for memory allocation.
    ge
        If True keep rows >= threshold (relay rightward); else keep rows
        <= threshold (relay leftward).

    Returns
    -------
    Filtered and repacked payload, or None if no rows survive the filter.
    """
    i64 = plc.DataType(plc.TypeId.INT64)
    bool8 = plc.DataType(plc.TypeId.BOOL8)
    table = unpack_and_concat(partitions=[pd], stream=stream, br=br)
    idx_col = _get_idx_col_i64(table, index_col_idx, type_id, i64, stream)
    if ge:
        filtered = _filter_ge(table, idx_col, threshold, i64, bool8, stream)
    else:
        filtered = _filter_le(table, idx_col, threshold, i64, bool8, stream)
    return _pack_table(filtered, stream, br)


def _pack_table(table: plc.Table, stream: Stream, br: Any) -> PackedData | None:
    """Pack a plc.Table into PackedData; return None if the table is empty."""
    if table.num_rows() == 0:
        return None
    return PackedData.from_cudf_packed_columns(
        pack(table, stream=stream, mr=br.device_mr),
        stream,
        br,
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


def _compute_send_halos(
    mids: list[int],
    sm: SpillableMessages,
    index_col_idx: int,
    type_id: plc.TypeId,
    lookback: int,
    lookahead: int,
    get_stream: Callable[[], Stream],
    br: Any,
) -> tuple[PackedData | None, PackedData | None, int | None, int | None]:
    """
    Walk boundary chunks to build the inter-rank halo payloads.

    send_right: walk backward from the last chunk.  The threshold is anchored
    to the last chunk's max index: rows with index >= (last_max - lookback) are
    needed by the right neighbor.  Walking stops once a chunk's max index falls
    below the threshold (no earlier chunks can contribute qualifying rows).

    send_left: symmetric walk forward from the first chunk with the threshold
    anchored to the first chunk's min index.

    Also returns ``(local_min, local_max)``: the min/max index values across all
    local chunks, used by the caller to decide whether multi-hop relay is needed.

    ``mids`` is updated in-place as chunks are extracted and re-inserted.
    All GPU operations on a chunk's table view complete *before* re-insertion
    so the spill manager cannot invalidate the view mid-operation.
    Each chunk's own stream is used for its GPU operations; ``get_stream`` is
    used only to join streams when combining filtered tables from multiple chunks.
    """
    n = len(mids)
    i64 = plc.DataType(plc.TypeId.INT64)
    bool8 = plc.DataType(plc.TypeId.BOOL8)

    local_max: int | None = None  # max index of last non-empty chunk
    local_min: int | None = None  # min index of first non-empty chunk

    send_right: PackedData | None = None
    if lookback > 0:
        threshold_r: int | None = None
        right_pairs: list[tuple[plc.Table, Stream]] = []
        for i in range(n - 1, -1, -1):
            chunk = TableChunk.from_message(sm.extract(mid=mids[i]))
            chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
            table = chunk.table_view()
            chunk_stream = chunk.stream
            if table.num_rows() == 0:
                mids[i] = sm.insert(Message(0, chunk))
                continue
            idx_col = _get_idx_col_i64(table, index_col_idx, type_id, i64, chunk_stream)
            mn, mx = _minmax_py(idx_col, i64, chunk_stream)
            if threshold_r is None:
                threshold_r = mx - lookback
                local_max = mx  # first iteration = last non-empty chunk
            should_stop = mx < threshold_r
            if not should_stop:
                right_pairs.insert(
                    0,
                    (
                        _filter_ge(
                            table, idx_col, threshold_r, i64, bool8, chunk_stream
                        ),
                        chunk_stream,
                    ),
                )
            mids[i] = sm.insert(Message(0, chunk))
            if should_stop:
                break
        if right_pairs:
            if len(right_pairs) == 1:
                combined, combined_stream = right_pairs[0]
            else:
                combined_stream = get_joined_cuda_stream(
                    get_stream, upstreams=[s for _, s in right_pairs]
                )
                combined = plc.concatenate.concatenate(
                    [t for t, _ in right_pairs], stream=combined_stream
                )
            send_right = _pack_table(combined, combined_stream, br)

    send_left: PackedData | None = None
    if lookahead > 0:
        threshold_l: int | None = None
        left_pairs: list[tuple[plc.Table, Stream]] = []
        for i in range(n):
            chunk = TableChunk.from_message(sm.extract(mid=mids[i]))
            chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
            table = chunk.table_view()
            chunk_stream = chunk.stream
            if table.num_rows() == 0:
                mids[i] = sm.insert(Message(0, chunk))
                continue
            idx_col = _get_idx_col_i64(table, index_col_idx, type_id, i64, chunk_stream)
            mn, mx = _minmax_py(idx_col, i64, chunk_stream)
            if threshold_l is None:
                threshold_l = mn + lookahead
                local_min = mn  # first iteration = first non-empty chunk
            should_stop = mn > threshold_l
            if not should_stop:
                left_pairs.append(
                    (
                        _filter_le(
                            table, idx_col, threshold_l, i64, bool8, chunk_stream
                        ),
                        chunk_stream,
                    )
                )
            mids[i] = sm.insert(Message(0, chunk))
            if should_stop:
                break
        if left_pairs:
            if len(left_pairs) == 1:
                combined, combined_stream = left_pairs[0]
            else:
                combined_stream = get_joined_cuda_stream(
                    get_stream, upstreams=[s for _, s in left_pairs]
                )
                combined = plc.concatenate.concatenate(
                    [t for t, _ in left_pairs], stream=combined_stream
                )
            send_left = _pack_table(combined, combined_stream, br)

    # If only one walk ran, scan the complementary boundary chunk for the missing bound
    if local_max is not None and local_min is None:
        # backward walk ran but not forward; peek at first non-empty chunk for local_min
        for j in range(n):
            chunk = TableChunk.from_message(sm.extract(mid=mids[j]))
            chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
            table = chunk.table_view()
            chunk_stream = chunk.stream
            if table.num_rows() > 0:
                idx_col = _get_idx_col_i64(
                    table, index_col_idx, type_id, i64, chunk_stream
                )
                local_min, _ = _minmax_py(idx_col, i64, chunk_stream)
                mids[j] = sm.insert(Message(0, chunk))
                break
            mids[j] = sm.insert(Message(0, chunk))
    elif local_min is not None and local_max is None:
        # forward walk ran but not backward; peek at last non-empty chunk for local_max
        for j in range(n - 1, -1, -1):
            chunk = TableChunk.from_message(sm.extract(mid=mids[j]))
            chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
            table = chunk.table_view()
            chunk_stream = chunk.stream
            if table.num_rows() > 0:
                idx_col = _get_idx_col_i64(
                    table, index_col_idx, type_id, i64, chunk_stream
                )
                _, local_max = _minmax_py(idx_col, i64, chunk_stream)
                mids[j] = sm.insert(Message(0, chunk))
                break
            mids[j] = sm.insert(Message(0, chunk))

    return send_left, send_right, local_min, local_max


async def _unpack_pds_to_table(
    pds: list[PackedData], stream: Stream, br: Any
) -> plc.Table | None:
    """Unpack a list of PackedData into a single plc.Table, or None if empty."""
    if not pds:
        return None
    return await asyncio.to_thread(
        unpack_and_concat, partitions=pds, stream=stream, br=br
    )


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
    """
    Streaming distributed rolling window actor.

    Algorithm
    ---------
    1. Buffer all incoming chunks as spillable messages.
    2. Walk boundary chunks to compute inter-rank halo payloads.
    3. HaloExchange loop (skipped for nranks==1).
    4. Output loop: for each local chunk, build left/right context from the
       sliding left-context accumulator and a lookahead peek at the next chunk,
       evaluate Rolling, strip context rows, apply zlice, and send one result
       chunk per input chunk.
    """
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

        # Window parameters
        type_id = ir.index_dtype.id()
        preceding_native = _ordinal_to_native(type_id, ir.preceding_ordinal)
        following_native = _ordinal_to_native(type_id, ir.following_ordinal)
        lookback = max(0, -preceding_native)
        lookahead = max(0, preceding_native + following_native)
        index_col_idx = col_names.index(ir.index.name)
        i64 = plc.DataType(plc.TypeId.INT64)
        bool8 = plc.DataType(plc.TypeId.BOOL8)

        # Inter-rank halo exchange (skipped for nranks == 1)
        halo_left_pds: list[PackedData] = []
        halo_right_pds: list[PackedData] = []

        if comm.nranks > 1:
            send_left, send_right, local_min, local_max = _compute_send_halos(
                mids,
                sm,
                index_col_idx,
                type_id,
                lookback,
                lookahead,
                ir_context.get_cuda_stream,  # factory; only used to join streams
                br,
            )

            halo_exchange_id = collective_ids.pop()
            allreduce_id = collective_ids.pop()
            he = HaloExchange(context, comm, halo_exchange_id)

            while True:
                from_left, from_right = await he.exchange(send_left, send_right)
                if from_left is not None:
                    halo_left_pds.insert(0, from_left)
                if from_right is not None:
                    halo_right_pds.append(from_right)

                # Relay with threshold filtering: only forward rows the next
                # rank actually needs.  The threshold mirrors the one used when
                # computing the initial send_right/send_left (anchored to this
                # rank's local_max/local_min).  If nothing survives the filter
                # the relay becomes None, terminating in O(lookback/range)
                # rounds rather than O(N) ranks.
                relay_stream = ir_context.get_cuda_stream()
                relay_right: PackedData | None = None
                if from_left is not None and local_max is not None:
                    relay_right = await asyncio.to_thread(
                        _filter_relay_pd,
                        from_left,
                        index_col_idx,
                        type_id,
                        local_max - lookback,
                        relay_stream,
                        br,
                        ge=True,
                    )
                relay_left: PackedData | None = None
                if from_right is not None and local_min is not None:
                    relay_left = await asyncio.to_thread(
                        _filter_relay_pd,
                        from_right,
                        index_col_idx,
                        type_id,
                        local_min + lookahead,
                        relay_stream,
                        br,
                        ge=False,
                    )

                my_done = (relay_right is None) and (relay_left is None)
                (total_done,) = await allgather_reduce(
                    context, comm, allreduce_id, int(my_done)
                )
                if total_done == comm.nranks:
                    break
                send_right = relay_right
                send_left = relay_left

        # Unpack inter-rank halos into DataFrames (carry their streams)
        halo_stream = ir_context.get_cuda_stream()
        left_halo_table = await _unpack_pds_to_table(halo_left_pds, halo_stream, br)
        right_halo_table = await _unpack_pds_to_table(halo_right_pds, halo_stream, br)

        left_ctx_df: DataFrame | None = (
            DataFrame.from_table(
                left_halo_table, col_names, col_dtypes, stream=halo_stream
            )
            if left_halo_table is not None and left_halo_table.num_rows() > 0
            else None
        )
        right_halo_df: DataFrame | None = (
            DataFrame.from_table(
                right_halo_table, col_names, col_dtypes, stream=halo_stream
            )
            if right_halo_table is not None and right_halo_table.num_rows() > 0
            else None
        )

        # Chunkwise evaluation loop
        non_child_args_no_zlice = ir._non_child_args[:-1] + (None,)
        n_processed = 0  # running tally of input rows processed (for zlice)
        for i, mid in enumerate(mids):
            chunk = TableChunk.from_message(sm.extract(mid=mid))
            chunk = chunk.make_available_and_spill(br, allow_overbooking=True)
            chunk_table = chunk.table_view()
            n_chunk_rows = chunk_table.num_rows()

            if n_chunk_rows == 0:
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
                next_chunk = TableChunk.from_message(sm.extract(mid=mids[i + 1]))
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
                    local_result_df.table, local_result_df.stream, exclusive_view=True
                )
                if tracer is not None:
                    tracer.add_chunk(table=result_chunk.table_view())
                await ch_out.send(context, Message(0, result_chunk))

        await ch_out.drain(context)


@generate_ir_sub_network.register(Rolling)
def _(
    ir: Rolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate streaming sub-network for a Rolling IR node."""
    nodes, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    assert len(collective_ids) == 2, (
        f"Rolling requires 2 collective IDs, got {len(collective_ids)}"
    )
    nodes[ir] = [
        rolling_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            ch_in=channels[ir.children[0]].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            collective_ids=collective_ids,
        )
    ]
    return nodes, channels
