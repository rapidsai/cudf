# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordering adjustment utilities for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc
from cudf_streaming.partition_utils import (
    packed_data_from_cudf_packed_columns,
    unpack_and_concat,
)
from cudf_streaming.table_chunk import TableChunk
from pylibcudf.contiguous_split import pack
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.streaming.actor_graph.utils import (
    ChunkStore,
    concat_batch,
    empty_table_chunk,
    gather_in_task_group,
)
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from cudf_streaming.channel_metadata import Ordering
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.memory.packed_data import PackedData
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext


_PID_DTYPE = DataType(pl.Int32())
_PID_PLC_DTYPE = plc.DataType(plc.TypeId.INT32)


def _contiguous_owner(pid: int, nranks: int, npartitions: int) -> int:
    """Return the rank owning *pid* under contiguous partition assignment."""
    return pid * nranks // npartitions


def _partition_range(rank: int, nranks: int, npartitions: int) -> tuple[int, int]:
    """Return the half-open partition ID range owned by *rank*."""
    return (
        (rank * npartitions + nranks - 1) // nranks,
        ((rank + 1) * npartitions + nranks - 1) // nranks,
    )


def _validate_orderings(input_ordering: Ordering, output_ordering: Ordering) -> None:
    """Validate the first-pass flat Ordering adjustment contract."""
    if not output_ordering.strict_boundaries:
        raise ValueError("adjust_ordering requires a strict output Ordering.")
    prefix_len = len(output_ordering.keys)
    if input_ordering.keys[:prefix_len] != output_ordering.keys:
        raise NotImplementedError(
            "adjust_ordering currently requires the output Ordering keys "
            "to be a prefix of the input Ordering keys."
        )


def _split_points(
    table: plc.Table,
    boundary_table: plc.Table,
    ordering: Ordering,
    stream: Stream,
) -> list[int]:
    """Return row split points that partition *table* by *ordering* boundaries."""
    if boundary_table.num_rows() == 0:
        return []
    key_table = plc.Table([table.columns()[key.column_index] for key in ordering.keys])
    split_col = plc.search.lower_bound(
        key_table,
        boundary_table,
        [key.order for key in ordering.keys],
        [key.null_order for key in ordering.keys],
        stream=stream,
    )
    return (
        DataFrame.from_table(
            plc.Table([split_col]),
            ["split"],
            [_PID_DTYPE],
            stream,
        )
        .to_polars()["split"]
        .to_list()
    )


def _append_partition_id(table: plc.Table, pid: int, stream: Stream) -> plc.Table:
    """Append a hidden target-partition-id column to *table*."""
    pid_col = plc.Column.from_scalar(
        plc.Scalar.from_py(pid, _PID_PLC_DTYPE, stream=stream),
        table.num_rows(),
        stream=stream,
    )
    return plc.Table([*table.columns(), pid_col])


def _boundary_search_positions(
    input_boundary_table: plc.Table,
    output_boundary_table: plc.Table,
    output_ordering: Ordering,
    stream: Stream,
) -> tuple[list[int], list[int]]:
    """Search output boundary positions for projected input boundary rows."""
    if input_boundary_table.num_rows() == 0:
        return [], []
    prefix_len = len(output_ordering.keys)
    input_prefix_boundaries = plc.Table(input_boundary_table.columns()[:prefix_len])
    orders = [key.order for key in output_ordering.keys]
    null_orders = [key.null_order for key in output_ordering.keys]
    lower_col = plc.search.lower_bound(
        output_boundary_table,
        input_prefix_boundaries,
        orders,
        null_orders,
        stream=stream,
    )
    upper_col = plc.search.upper_bound(
        output_boundary_table,
        input_prefix_boundaries,
        orders,
        null_orders,
        stream=stream,
    )
    positions = DataFrame.from_table(
        plc.Table([lower_col, upper_col]),
        ["lower", "upper"],
        [_PID_DTYPE, _PID_DTYPE],
        stream,
    ).to_polars()
    return positions["lower"].to_list(), positions["upper"].to_list()


def _source_output_range(
    source_rank: int,
    nranks: int,
    input_ordering: Ordering,
    output_ordering: Ordering,
    lower_positions: list[int],
    upper_positions: list[int],
) -> tuple[int, int]:
    """Return the half-open output partition range touched by a source rank."""
    input_npartitions = input_ordering.num_boundaries + 1
    output_npartitions = output_ordering.num_boundaries + 1
    output_prefix_only = len(output_ordering.keys) < len(input_ordering.keys)
    include_upper_boundary = output_prefix_only or not input_ordering.strict_boundaries
    input_start, input_stop = _partition_range(source_rank, nranks, input_npartitions)
    if input_start == input_stop:
        return 0, 0
    output_start = 0 if input_start == 0 else upper_positions[input_start - 1]
    output_stop = (
        output_npartitions
        if input_stop == input_npartitions
        else (
            upper_positions[input_stop - 1] + 1
            if include_upper_boundary
            else lower_positions[input_stop - 1] + 1
        )
    )
    return output_start, output_stop


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Return whether two half-open integer ranges overlap."""
    return max(left[0], right[0]) < min(left[1], right[1])


def _unpack_remote_piece(
    packed: PackedData,
    stream: Stream,
    br: BufferResource,
) -> tuple[int, TableChunk] | None:
    """Unpack one remote piece and recover its hidden target partition ID."""
    table = unpack_and_concat([packed], stream=stream, br=br)
    if table.num_rows() == 0:
        return None
    *payload_cols, pid_col = table.columns()
    pid = int(
        DataFrame.from_table(
            plc.Table([pid_col]),
            ["pid"],
            [_PID_DTYPE],
            stream,
        )
        .to_polars()
        .item(0, 0)
    )
    payload = plc.concatenate.concatenate(
        [plc.Table(payload_cols)], stream=stream, mr=br.device_mr
    )
    return pid, TableChunk.from_pylibcudf_table(
        payload,
        stream,
        exclusive_view=True,
        br=br,
    )


def _copy_to_owned_chunk(
    table: plc.Table,
    stream: Stream,
    br: BufferResource,
) -> TableChunk:
    """Copy a table view into a uniquely-owned chunk."""
    table = plc.concatenate.concatenate([table], stream=stream, mr=br.device_mr)
    return TableChunk.from_pylibcudf_table(
        table,
        stream,
        exclusive_view=True,
        br=br,
    )


class _OutputPieceReader:
    """Read input only far enough to materialize requested output windows."""

    def __init__(
        self,
        context: Context,
        ch_in: Channel[TableChunk],
        boundary_chunk: TableChunk,
        output_ordering: Ordering,
    ) -> None:
        self.context = context
        self.ch_in = ch_in
        self.boundary_chunk = boundary_chunk
        self.boundary_table = boundary_chunk.table_view()
        self.output_ordering = output_ordering
        self.pending: dict[int, ChunkStore] = {}
        self.input_done = False

    def _store(self, pid: int, chunk: TableChunk) -> None:
        if pid not in self.pending:
            self.pending[pid] = ChunkStore(self.context)
        self.pending[pid].insert(Message(pid, chunk))

    def _has_reached(self, stop: int) -> bool:
        return any(pid >= stop for pid in self.pending)

    async def collect_window(self, start: int, stop: int) -> dict[int, ChunkStore]:
        """Return all locally-read pieces for output pids in ``[start, stop)``."""
        while not self.input_done and not self._has_reached(stop):
            msg = await self.ch_in.recv(self.context)
            if msg is None:
                self.input_done = True
                break
            chunk = TableChunk.from_message(
                msg, br=self.context.br()
            ).make_available_and_spill(self.context.br(), allow_overbooking=True)
            if chunk.table_view().num_rows() == 0:
                continue
            with stream_ordered_after(
                self.context.br().stream_pool.get_stream,
                upstreams=(chunk.stream, self.boundary_chunk.stream),
            ) as stream:
                table = chunk.table_view()
                splits = _split_points(
                    table, self.boundary_table, self.output_ordering, stream
                )
                for pid, piece in enumerate(
                    plc.copying.split(table, splits, stream=stream)
                ):
                    if piece.num_rows() == 0:
                        continue
                    self._store(
                        pid, _copy_to_owned_chunk(piece, stream, self.context.br())
                    )

        out = {
            pid: self.pending.pop(pid)
            for pid in list(self.pending)
            if start <= pid < stop
        }
        return dict(sorted(out.items()))


async def _adjust_ordering_local(
    context: Context,
    ref_ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    output_ordering: Ordering,
) -> None:
    npartitions = output_ordering.num_boundaries + 1
    boundary_chunk = output_ordering.get_boundaries(context.br())
    boundary_table = boundary_chunk.table_view()
    pending_pid: int | None = None
    pending_chunks: ChunkStore | None = None
    next_pid = 0

    async def emit_pending(pid: int) -> None:
        nonlocal pending_pid, pending_chunks
        if pending_pid == pid and pending_chunks is not None:
            chunks = [
                TableChunk.from_message(msg, br=context.br()) for msg in pending_chunks
            ]
            chunk = await concat_batch(chunks, context, ref_ir.schema, ir_context)
            pending_pid = None
            pending_chunks = None
        else:
            chunk = empty_table_chunk(ref_ir, context, ir_context.get_cuda_stream())
        await ch_out.send(context, Message(pid, chunk))

    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg, br=context.br()).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        if chunk.table_view().num_rows() == 0:
            continue
        with stream_ordered_after(
            context.br().stream_pool.get_stream,
            upstreams=(chunk.stream, boundary_chunk.stream),
        ) as stream:
            table = chunk.table_view()
            splits = _split_points(table, boundary_table, output_ordering, stream)
            for pid, piece in enumerate(
                plc.copying.split(table, splits, stream=stream)
            ):
                if piece.num_rows() == 0:
                    continue
                if pending_pid is not None and pending_pid != pid:
                    emitted_pid = pending_pid
                    await emit_pending(emitted_pid)
                    next_pid = emitted_pid + 1
                while next_pid < pid:
                    await emit_pending(next_pid)
                    next_pid += 1
                if pending_pid is None:
                    pending_pid = pid
                    pending_chunks = ChunkStore(context)
                assert pending_chunks is not None
                pending_chunks.insert(
                    Message(
                        pid,
                        _copy_to_owned_chunk(piece, stream, context.br()),
                    )
                )

    while next_pid < npartitions:
        await emit_pending(next_pid)
        next_pid += 1
    await ch_out.drain(context)


def _store_chunk(
    context: Context,
    stores: dict[int, ChunkStore],
    pid: int,
    chunk: TableChunk,
) -> None:
    if pid not in stores:
        stores[pid] = ChunkStore(context)
    stores[pid].insert(Message(pid, chunk))


async def _adjust_ordering_rank_hybrid(
    context: Context,
    comm: Communicator,
    ref_ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    input_ordering: Ordering,
    output_ordering: Ordering,
    collective_id: int,
    lower_positions: list[int],
    upper_positions: list[int],
) -> None:
    npartitions = output_ordering.num_boundaries + 1
    boundary_chunk = output_ordering.get_boundaries(context.br())
    boundary_table = boundary_chunk.table_view()
    local_window = _partition_range(comm.rank, comm.nranks, npartitions)
    source_ranges = [
        _source_output_range(
            source_rank,
            comm.nranks,
            input_ordering,
            output_ordering,
            lower_positions,
            upper_positions,
        )
        for source_rank in range(comm.nranks)
    ]
    local_source_range = source_ranges[comm.rank]
    srcs = [
        source_rank
        for source_rank, source_range in enumerate(source_ranges)
        if source_rank != comm.rank and _ranges_overlap(source_range, local_window)
    ]
    dsts = [
        output_rank
        for output_rank in range(comm.nranks)
        if output_rank != comm.rank
        and _ranges_overlap(
            local_source_range, _partition_range(output_rank, comm.nranks, npartitions)
        )
    ]
    exchange = SparseAlltoall(
        context,
        comm,
        collective_id,
        srcs=srcs,
        dsts=dsts,
    )

    async def send_piece(pid: int, chunk: TableChunk) -> None:
        with stream_ordered_after(
            context.br().stream_pool.get_stream,
            upstreams=(chunk.stream,),
        ) as stream:
            exchange.insert(
                _contiguous_owner(pid, comm.nranks, npartitions),
                packed_data_from_cudf_packed_columns(
                    pack(
                        _append_partition_id(chunk.table_view(), pid, stream),
                        stream,
                        mr=context.br().device_mr,
                    ),
                    stream,
                    context.br(),
                ),
            )

    async def emit(pid: int, store: ChunkStore | None) -> None:
        chunks = (
            [TableChunk.from_message(msg, br=context.br()) for msg in store]
            if store is not None
            else []
        )
        chunk = (
            await concat_batch(chunks, context, ref_ir.schema, ir_context)
            if chunks
            else empty_table_chunk(ref_ir, context, ir_context.get_cuda_stream())
        )
        await ch_out.send(context, Message(pid, chunk))

    # Ranks with no incoming dependency can stream local output immediately
    # while sending remote-owned pieces as they are encountered.
    if not srcs:
        pending_pid: int | None = None
        pending_chunks: ChunkStore | None = None
        next_pid = local_window[0]

        async def emit_pending(pid: int) -> None:
            nonlocal pending_pid, pending_chunks
            await emit(pid, pending_chunks if pending_pid == pid else None)
            if pending_pid == pid:
                pending_pid = None
                pending_chunks = None

        try:
            while (msg := await ch_in.recv(context)) is not None:
                chunk = TableChunk.from_message(
                    msg, br=context.br()
                ).make_available_and_spill(context.br(), allow_overbooking=True)
                if chunk.table_view().num_rows() == 0:
                    continue
                with stream_ordered_after(
                    context.br().stream_pool.get_stream,
                    upstreams=(chunk.stream, boundary_chunk.stream),
                ) as stream:
                    table = chunk.table_view()
                    splits = _split_points(
                        table, boundary_table, output_ordering, stream
                    )
                    for pid, piece in enumerate(
                        plc.copying.split(table, splits, stream=stream)
                    ):
                        if piece.num_rows() == 0:
                            continue
                        owner = _contiguous_owner(pid, comm.nranks, npartitions)
                        owned = owner == comm.rank
                        piece_chunk = _copy_to_owned_chunk(piece, stream, context.br())
                        if not owned:
                            await send_piece(pid, piece_chunk)
                            continue
                        if pending_pid is not None and pending_pid != pid:
                            emitted_pid = pending_pid
                            await emit_pending(emitted_pid)
                            next_pid = emitted_pid + 1
                        while next_pid < pid:
                            await emit_pending(next_pid)
                            next_pid += 1
                        if pending_pid is None:
                            pending_pid = pid
                            pending_chunks = ChunkStore(context)
                        assert pending_chunks is not None
                        pending_chunks.insert(Message(pid, piece_chunk))
        finally:
            await exchange.insert_finished(context)

        while next_pid < local_window[1]:
            await emit_pending(next_pid)
            next_pid += 1
        await ch_out.drain(context)
        return

    reader = _OutputPieceReader(context, ch_in, boundary_chunk, output_ordering)
    local_pieces: dict[int, ChunkStore] = {}
    # If a higher rank depends on this rank's data, read far enough to make
    # that data available before waiting for lower-rank input.
    if dsts:
        pieces = await reader.collect_window(*local_source_range)
        for pid, store in pieces.items():
            owner = _contiguous_owner(pid, comm.nranks, npartitions)
            if owner == comm.rank:
                local_pieces[pid] = store
                continue
            for msg in store:
                await send_piece(pid, TableChunk.from_message(msg, br=context.br()))
    await exchange.insert_finished(context)

    pieces_by_source: dict[int, dict[int, ChunkStore]] = {}
    if local_pieces:
        pieces_by_source[comm.rank] = local_pieces
    for source_rank in srcs:
        remote_pieces: dict[int, ChunkStore] = {}
        stream = context.br().stream_pool.get_stream()
        for packed in exchange.extract(source_rank):
            remote_piece = _unpack_remote_piece(packed, stream, context.br())
            if remote_piece is None:
                continue
            pid, chunk = remote_piece
            _store_chunk(context, remote_pieces, pid, chunk)
        pieces_by_source[source_rank] = remote_pieces

    for pid, store in (await reader.collect_window(*local_window)).items():
        if _contiguous_owner(pid, comm.nranks, npartitions) == comm.rank:
            if pid not in local_pieces:
                local_pieces[pid] = ChunkStore(context)
            for msg in store:
                local_pieces[pid].insert(msg)
    if local_pieces:
        pieces_by_source[comm.rank] = local_pieces

    contributing_sources = [
        source_rank
        for source_rank, source_range in enumerate(source_ranges)
        if _ranges_overlap(source_range, local_window)
    ]
    for pid in range(*local_window):
        chunks: list[TableChunk] = []
        for source_rank in contributing_sources:
            stores = pieces_by_source.get(source_rank)
            if stores is None:
                continue
            pid_store = stores.get(pid)
            if pid_store is None:
                continue
            chunks.extend(
                TableChunk.from_message(msg, br=context.br()) for msg in pid_store
            )
        chunk = (
            await concat_batch(chunks, context, ref_ir.schema, ir_context)
            if chunks
            else empty_table_chunk(ref_ir, context, ir_context.get_cuda_stream())
        )
        await ch_out.send(context, Message(pid, chunk))
    await ch_out.drain(context)


async def adjust_ordering(
    context: Context,
    comm: Communicator,
    ref_ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    input_ordering: Ordering,
    output_ordering: Ordering,
    *,
    collective_id: int | None = None,
) -> None:
    """
    Adjust flat Ordering boundaries using contiguous partition ownership.

    Parameters
    ----------
    context
        The streaming context.
    comm
        The communicator.
    ref_ir
        An IR node describing the payload schema.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    input_ordering
        The input Ordering.
    output_ordering
        The output Ordering.
    collective_id
        The collective ID to use for SparseAlltoall.

    Notes
    -----
    This utility is intentionally narrow and only adjusts data messages. The
    caller is responsible for receiving input metadata and sending output
    metadata. Input rows are assumed to be globally ordered by ``input_ordering``;
    sortedness is not checked here.
    """
    _validate_orderings(input_ordering, output_ordering)

    if comm.nranks > 1 and collective_id is None:
        raise ValueError("collective_id is required when comm.nranks > 1.")

    try:
        if comm.nranks == 1:
            await _adjust_ordering_local(
                context,
                ref_ir,
                ir_context,
                ch_out,
                ch_in,
                output_ordering,
            )
            return

        input_boundary_chunk = input_ordering.get_boundaries(context.br())
        boundary_chunk = output_ordering.get_boundaries(context.br())
        boundary_table = boundary_chunk.table_view()
        with stream_ordered_after(
            context.br().stream_pool.get_stream,
            upstreams=(input_boundary_chunk.stream, boundary_chunk.stream),
        ) as stream:
            lower_positions, upper_positions = _boundary_search_positions(
                input_boundary_chunk.table_view(),
                boundary_table,
                output_ordering,
                stream,
            )
        assert collective_id is not None
        await _adjust_ordering_rank_hybrid(
            context,
            comm,
            ref_ir,
            ir_context,
            ch_out,
            ch_in,
            input_ordering,
            output_ordering,
            collective_id,
            lower_positions,
            upper_positions,
        )
    except BaseException:
        await gather_in_task_group(
            ch_in.shutdown(context),
            ch_in.shutdown_metadata(context),
            ch_out.shutdown(context),
            ch_out.shutdown_metadata(context),
        )
        raise
