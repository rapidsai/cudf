# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OrderScheme adjustment utilities for the RapidsMPF streaming runtime."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc
from cudf_streaming.partition_utils import unpack_and_concat
from cudf_streaming.table_chunk import TableChunk
from pylibcudf.contiguous_split import pack
from rapidsmpf.memory.packed_data import PackedData
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
    from cudf_streaming.channel_metadata import OrderScheme, Ordering
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext


_PID_DTYPE = DataType(pl.Int32())
_PID_PLC_DTYPE = plc.DataType(plc.TypeId.INT32)


def _primary_ordering(scheme: OrderScheme) -> Ordering:
    """Return the single ordering supported by adjust_orderscheme for now."""
    (ordering,) = scheme.orderings
    return ordering


def _contiguous_owner(pid: int, nranks: int, npartitions: int) -> int:
    """Return the rank owning *pid* under contiguous partition assignment."""
    return pid * nranks // npartitions


def _partition_range(rank: int, nranks: int, npartitions: int) -> tuple[int, int]:
    """Return the half-open partition ID range owned by *rank*."""
    return (
        (rank * npartitions + nranks - 1) // nranks,
        ((rank + 1) * npartitions + nranks - 1) // nranks,
    )


def _local_partitions(rank: int, nranks: int, npartitions: int) -> list[int]:
    """Return partition IDs owned by *rank* under contiguous assignment."""
    start, stop = _partition_range(rank, nranks, npartitions)
    return list(range(start, stop))


def _contiguous_owners(
    start: int,
    stop: int,
    nranks: int,
    npartitions: int,
) -> list[int]:
    """Return ranks owning any partition in the half-open range [start, stop)."""
    if start >= stop:
        return []
    first_rank = _contiguous_owner(start, nranks, npartitions)
    last_rank = _contiguous_owner(stop - 1, nranks, npartitions)
    owners = []
    for rank in range(first_rank, last_rank + 1):
        rank_start, rank_stop = _partition_range(rank, nranks, npartitions)
        if max(start, rank_start) < min(stop, rank_stop):
            owners.append(rank)
    return owners


def _validate_schemes(input_scheme: OrderScheme, output_scheme: OrderScheme) -> None:
    """Validate the first-pass flat OrderScheme adjustment contract."""
    input_ordering = _primary_ordering(input_scheme)
    output_ordering = _primary_ordering(output_scheme)
    if not output_ordering.strict_boundaries:
        raise ValueError("adjust_orderscheme requires a strict output OrderScheme.")
    prefix_len = len(output_ordering.keys)
    if input_ordering.keys[:prefix_len] != output_ordering.keys:
        raise NotImplementedError(
            "adjust_orderscheme currently requires the output OrderScheme keys "
            "to be a prefix of the input OrderScheme keys."
        )


def _split_points(
    table: plc.Table,
    boundary_table: plc.Table,
    ordering: Ordering,
    stream: Stream,
) -> list[int]:
    """Return row split points that partition *table* by *scheme* boundaries."""
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


def _peer_ranks(
    rank: int,
    nranks: int,
    input_ordering: Ordering,
    output_ordering: Ordering,
    lower_positions: list[int],
    upper_positions: list[int],
) -> tuple[list[int], list[int]]:
    """Return source and destination ranks needed for OrderScheme adjustment."""
    input_npartitions = input_ordering.num_boundaries + 1
    output_npartitions = output_ordering.num_boundaries + 1
    output_prefix_only = len(output_ordering.keys) < len(input_ordering.keys)
    include_upper_boundary = output_prefix_only or not input_ordering.strict_boundaries

    def dsts_for_source(source_rank: int) -> list[int]:
        input_start, input_stop = _partition_range(
            source_rank, nranks, input_npartitions
        )
        if input_start == input_stop:
            return []
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
        return [
            dst
            for dst in _contiguous_owners(
                output_start, output_stop, nranks, output_npartitions
            )
            if dst != source_rank
        ]

    dsts = dsts_for_source(rank)
    srcs = [
        source_rank
        for source_rank in range(nranks)
        if source_rank != rank and rank in dsts_for_source(source_rank)
    ]
    return srcs, dsts


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


async def _adjust_orderscheme_local(
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


async def adjust_orderscheme(
    context: Context,
    comm: Communicator,
    ref_ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    input_scheme: OrderScheme,
    output_scheme: OrderScheme,
    *,
    collective_id: int | None = None,
) -> None:
    """
    Adjust flat OrderScheme boundaries using contiguous partition ownership.

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
    input_scheme
        The input OrderScheme.
    output_scheme
        The output OrderScheme.
    collective_id
        The collective ID to use for SparseAlltoall.

    Notes
    -----
    This utility is intentionally narrow and only adjusts data messages. The
    caller is responsible for receiving input metadata and sending output
    metadata. Input rows are assumed to be globally ordered by ``input_scheme``;
    sortedness is not checked here.
    """
    _validate_schemes(input_scheme, output_scheme)
    input_ordering = _primary_ordering(input_scheme)
    output_ordering = _primary_ordering(output_scheme)
    npartitions = output_ordering.num_boundaries + 1
    local_pids = _local_partitions(comm.rank, comm.nranks, npartitions)

    if comm.nranks > 1 and collective_id is None:
        raise ValueError("collective_id is required when comm.nranks > 1.")

    try:
        if comm.nranks == 1:
            await _adjust_orderscheme_local(
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
        srcs: list[int] = []
        dsts: list[int] = []
        if comm.nranks > 1:
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
            srcs, dsts = _peer_ranks(
                comm.rank,
                comm.nranks,
                input_ordering,
                output_ordering,
                lower_positions,
                upper_positions,
            )
        exchange = (
            SparseAlltoall(context, comm, collective_id, srcs=srcs, dsts=dsts)
            if comm.nranks > 1
            else None
        )
        local_chunks: dict[int, list[TableChunk]] = defaultdict(list)

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
                        if owner == comm.rank:
                            local_chunks[pid].append(
                                _copy_to_owned_chunk(piece, stream, context.br())
                            )
                        else:
                            assert exchange is not None
                            exchange.insert(
                                owner,
                                PackedData.from_cudf_packed_columns(
                                    pack(
                                        _append_partition_id(piece, pid, stream),
                                        stream,
                                        mr=context.br().device_mr,
                                    ),
                                    stream,
                                    context.br(),
                                ),
                            )
        finally:
            if exchange is not None:
                await exchange.insert_finished(context)

        output_chunks: dict[int, list[TableChunk]] = defaultdict(list)
        for source_rank in (
            *[src for src in srcs if src < comm.rank],
            comm.rank,
            *[src for src in srcs if src > comm.rank],
        ):
            if source_rank == comm.rank:
                for pid, chunks in local_chunks.items():
                    output_chunks[pid].extend(chunks)
            else:
                assert exchange is not None
                stream = context.br().stream_pool.get_stream()
                for packed in exchange.extract(source_rank):
                    remote_piece = _unpack_remote_piece(packed, stream, context.br())
                    if remote_piece is None:
                        continue
                    pid, chunk = remote_piece
                    output_chunks[pid].append(chunk)

        for pid in local_pids:
            chunks = output_chunks[pid]
            chunk = (
                await concat_batch(chunks, context, ref_ir.schema, ir_context)
                if chunks
                else empty_table_chunk(ref_ir, context, ir_context.get_cuda_stream())
            )
            await ch_out.send(context, Message(pid, chunk))
        await ch_out.drain(context)
    except BaseException:
        await gather_in_task_group(
            ch_in.shutdown(context),
            ch_in.shutdown_metadata(context),
            ch_out.shutdown(context),
            ch_out.shutdown_metadata(context),
        )
        raise
