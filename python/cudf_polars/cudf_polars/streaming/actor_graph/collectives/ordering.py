# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adjust streams between concrete Ordering boundary layouts without sorting."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class _RoutingPlan:
    npartitions: int
    boundary_chunk: TableChunk
    local_window: tuple[int, int]
    source_ranges: list[tuple[int, int]]
    remote_sources: list[int]
    owed_remote_pids: list[int]
    remote_destinations: list[int]
    owed_remote_range: tuple[int, int]


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
    """Validate the Ordering pair supported by this data-movement primitive."""
    if not output_ordering.strict_boundaries:
        raise ValueError("adjust_ordering requires strict output boundaries.")
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
    """Append a temporary target-partition-id column to *table*."""
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
    # Keep both sides of equal-boundary runs for prefix/non-strict cases.
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
    # Prefix/non-strict boundaries can overlap the equal-boundary run above them.
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


def _sources_for_pid(
    source_ranges: list[tuple[int, int]],
    pid: int,
) -> list[int]:
    """Return source ranks that may contribute to one output partition."""
    return [
        source_rank
        for source_rank, source_range in enumerate(source_ranges)
        if source_range[0] <= pid < source_range[1]
    ]


def _unpack_remote_piece(
    packed: PackedData,
    stream: Stream,
    br: BufferResource,
) -> tuple[int, TableChunk] | None:
    """Unpack one remote piece and recover its temporary target partition ID."""
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
    if not payload_cols:
        return None
    payload = plc.Table(payload_cols).copy(stream=stream, mr=br.device_mr)
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
    table = table.copy(stream=stream, mr=br.device_mr)
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
        self.output_ordering = output_ordering
        self.pending: dict[int, ChunkStore] = {}
        self.input_done = False

    async def collect_window(self, start: int, stop: int) -> dict[int, ChunkStore]:
        """Return all locally-read pieces for output pids in ``[start, stop)``."""
        if start >= stop:
            return {}
        while not self.input_done and not any(pid >= stop for pid in self.pending):
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
                    table,
                    self.boundary_chunk.table_view(),
                    self.output_ordering,
                    stream,
                )
                for pid, piece in enumerate(
                    plc.copying.split(table, splits, stream=stream)
                ):
                    if piece.num_rows() == 0:
                        continue
                    _store_chunk(
                        self.context,
                        self.pending,
                        pid,
                        _copy_to_owned_chunk(piece, stream, self.context.br()),
                    )

        out = {
            pid: self.pending.pop(pid)
            for pid in list(self.pending)
            if start <= pid < stop
        }
        return dict(sorted(out.items()))

    async def drain_remaining(self) -> None:
        """Consume input that cannot affect any remaining output partition."""
        while not self.input_done:
            if await self.ch_in.recv(self.context) is None:
                self.input_done = True


def _store_chunk(
    context: Context,
    stores: dict[int, ChunkStore],
    pid: int,
    chunk: TableChunk,
) -> None:
    if pid not in stores:
        stores[pid] = ChunkStore(context)
    stores[pid].insert(Message(pid, chunk))


async def _send_remote_store(
    context: Context,
    comm: Communicator,
    exchange: SparseAlltoall,
    npartitions: int,
    pid: int,
    store: ChunkStore,
) -> None:
    """Send all locally-read pieces for one remote-owned output partition."""
    for msg in store:
        await _send_remote_piece(
            context,
            comm,
            exchange,
            npartitions,
            pid,
            TableChunk.from_message(msg, br=context.br()),
        )


async def _send_remote_piece(
    context: Context,
    comm: Communicator,
    exchange: SparseAlltoall,
    npartitions: int,
    pid: int,
    chunk: TableChunk,
) -> None:
    """Send one output-partition piece to its remote owner."""
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


async def _send_remote_marker(
    context: Context,
    comm: Communicator,
    exchange: SparseAlltoall,
    npartitions: int,
    pid: int,
) -> None:
    """Send a no-payload marker for an empty remote-owned partition."""
    with stream_ordered_after(
        context.br().stream_pool.get_stream,
        upstreams=(),
    ) as stream:
        pid_col = plc.Column.from_scalar(
            plc.Scalar.from_py(pid, _PID_PLC_DTYPE, stream=stream),
            1,
            stream=stream,
        )
        exchange.insert(
            _contiguous_owner(pid, comm.nranks, npartitions),
            packed_data_from_cudf_packed_columns(
                pack(plc.Table([pid_col]), stream, mr=context.br().device_mr),
                stream,
                context.br(),
            ),
        )


async def _send_missing_remote_markers(
    context: Context,
    comm: Communicator,
    exchange: SparseAlltoall,
    npartitions: int,
    owed_remote_pids: list[int],
    sent_remote_pids: set[int],
) -> None:
    """Send empty markers for remote-owned partitions with no payload."""
    for pid in owed_remote_pids:
        if pid not in sent_remote_pids:
            await _send_remote_marker(context, comm, exchange, npartitions, pid)


async def _emit_partition(
    context: Context,
    ref_ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    pid: int,
    store: ChunkStore | None,
) -> None:
    """Emit one output partition, using an empty chunk when no data is present."""
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


def _make_routing_plan(
    context: Context,
    comm: Communicator,
    input_ordering: Ordering,
    output_ordering: Ordering,
) -> _RoutingPlan:
    """Compute local ownership and sparse-exchange obligations."""
    npartitions = output_ordering.num_boundaries + 1
    boundary_chunk = output_ordering.get_boundaries(context.br())
    local_window = _partition_range(comm.rank, comm.nranks, npartitions)

    if comm.nranks == 1:
        source_ranges = [(0, npartitions)]
    else:
        input_boundary_chunk = input_ordering.get_boundaries(context.br())
        with stream_ordered_after(
            context.br().stream_pool.get_stream,
            upstreams=(input_boundary_chunk.stream, boundary_chunk.stream),
        ) as stream:
            lower_positions, upper_positions = _boundary_search_positions(
                input_boundary_chunk.table_view(),
                boundary_chunk.table_view(),
                output_ordering,
                stream,
            )
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
    remote_sources = [
        source_rank
        for source_rank, source_range in enumerate(source_ranges)
        if source_rank != comm.rank and _ranges_overlap(source_range, local_window)
    ]
    owed_remote_pids = [
        pid
        for pid in range(*local_source_range)
        if _contiguous_owner(pid, comm.nranks, npartitions) != comm.rank
    ]
    remote_destinations = sorted(
        {_contiguous_owner(pid, comm.nranks, npartitions) for pid in owed_remote_pids}
    )
    owed_remote_range = (
        (owed_remote_pids[0], owed_remote_pids[-1] + 1) if owed_remote_pids else (0, 0)
    )
    return _RoutingPlan(
        npartitions=npartitions,
        boundary_chunk=boundary_chunk,
        local_window=local_window,
        source_ranges=source_ranges,
        remote_sources=remote_sources,
        owed_remote_pids=owed_remote_pids,
        remote_destinations=remote_destinations,
        owed_remote_range=owed_remote_range,
    )


async def _adjust_ordering_impl(
    context: Context,
    comm: Communicator,
    ref_ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    input_ordering: Ordering,
    output_ordering: Ordering,
    collective_id: int | None,
) -> None:
    """Adjust ordering while using exchange only for remote dependencies."""
    plan = _make_routing_plan(context, comm, input_ordering, output_ordering)
    reader = _OutputPieceReader(context, ch_in, plan.boundary_chunk, output_ordering)

    exchange = None
    if plan.remote_sources or plan.remote_destinations:
        assert collective_id is not None
        exchange = SparseAlltoall(
            context,
            comm,
            collective_id,
            srcs=plan.remote_sources,
            dsts=plan.remote_destinations,
        )
    local_pieces: dict[int, ChunkStore] = {}
    sent_remote_pids: set[int] = set()
    first_blocked_pid = next(
        (
            pid
            for pid in range(*plan.local_window)
            if any(
                source_rank != comm.rank
                for source_rank in _sources_for_pid(plan.source_ranges, pid)
            )
        ),
        plan.local_window[1],
    )

    pre_start, pre_stop = plan.local_window[0], first_blocked_pid
    if plan.owed_remote_pids:
        pre_start = min(pre_start, plan.owed_remote_range[0])
        pre_stop = max(pre_stop, plan.owed_remote_range[1])

    # Before receiving remote pieces, emit the local-only prefix and send all
    # remote-owned pieces this rank is responsible for.
    for pid in range(pre_start, pre_stop):
        pieces = await reader.collect_window(pid, pid + 1)
        owner = _contiguous_owner(pid, comm.nranks, plan.npartitions)
        if exchange is not None and owner != comm.rank and pid in pieces:
            sent_remote_pids.add(pid)
            await _send_remote_store(
                context,
                comm,
                exchange,
                plan.npartitions,
                pid,
                pieces[pid],
            )
        elif owner == comm.rank and pid in pieces:
            local_pieces[pid] = pieces[pid]
        if plan.local_window[0] <= pid < first_blocked_pid:
            await _emit_partition(
                context,
                ref_ir,
                ir_context,
                ch_out,
                pid,
                local_pieces.pop(pid, None),
            )

    if exchange is not None:
        await _send_missing_remote_markers(
            context,
            comm,
            exchange,
            plan.npartitions,
            plan.owed_remote_pids,
            sent_remote_pids,
        )
        await exchange.insert_finished(context)

    pieces_by_source: dict[int, dict[int, ChunkStore]] = {}
    pieces_by_source[comm.rank] = local_pieces
    if exchange is not None:
        for source_rank in plan.remote_sources:
            remote_pieces: dict[int, ChunkStore] = {}
            stream = context.br().stream_pool.get_stream()
            for packed in exchange.extract(source_rank):
                remote_piece = _unpack_remote_piece(packed, stream, context.br())
                if remote_piece is None:
                    continue
                pid, chunk = remote_piece
                _store_chunk(context, remote_pieces, pid, chunk)
            pieces_by_source[source_rank] = remote_pieces

    for pid in range(first_blocked_pid, plan.local_window[1]):
        pid_sources = _sources_for_pid(plan.source_ranges, pid)
        if comm.rank in pid_sources and pid not in local_pieces:
            local_pieces.update(
                {
                    piece_pid: store
                    for piece_pid, store in (
                        await reader.collect_window(pid, pid + 1)
                    ).items()
                    if _contiguous_owner(piece_pid, comm.nranks, plan.npartitions)
                    == comm.rank
                }
            )
        chunks: list[TableChunk] = []
        for source_rank in pid_sources:
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
    await reader.drain_remaining()
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
        await _adjust_ordering_impl(
            context,
            comm,
            ref_ir,
            ir_context,
            ch_out,
            ch_in,
            input_ordering,
            output_ordering,
            collective_id,
        )
    except BaseException:
        await gather_in_task_group(
            ch_in.shutdown(context),
            ch_in.shutdown_metadata(context),
            ch_out.shutdown(context),
            ch_out.shutdown_metadata(context),
        )
        raise
