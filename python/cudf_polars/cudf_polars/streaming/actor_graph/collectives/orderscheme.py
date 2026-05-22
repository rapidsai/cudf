# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""OrderScheme adjustment utilities for the RapidsMPF streaming runtime."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import pylibcudf as plc
from pylibcudf.contiguous_split import pack

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.streaming.actor_graph.utils import concat_batch, empty_table_chunk
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.channel_metadata import OrderScheme

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext


_PID_DTYPE = DataType(pl.Int32())
_PID_PLC_DTYPE = plc.DataType(plc.TypeId.INT32)


def _contiguous_owner(pid: int, nranks: int, npartitions: int) -> int:
    """Return the rank owning *pid* under contiguous partition assignment."""
    return pid * nranks // npartitions


def _local_partitions(rank: int, nranks: int, npartitions: int) -> list[int]:
    """Return partition IDs owned by *rank* under contiguous assignment."""
    return [
        pid
        for pid in range(npartitions)
        if _contiguous_owner(pid, nranks, npartitions) == rank
    ]


def _validate_schemes(input_scheme: OrderScheme, output_scheme: OrderScheme) -> None:
    """Validate the first-pass flat OrderScheme adjustment contract."""
    if not output_scheme.strict_boundaries:
        raise ValueError("adjust_orderscheme requires a strict output OrderScheme.")
    prefix_len = len(output_scheme.keys)
    if input_scheme.keys[:prefix_len] != output_scheme.keys:
        raise NotImplementedError(
            "adjust_orderscheme currently requires the output OrderScheme keys "
            "to be a prefix of the input OrderScheme keys."
        )


def _split_points(
    table: plc.Table,
    boundary_table: plc.Table,
    scheme: OrderScheme,
    stream: Stream,
) -> list[int]:
    """Return row split points that partition *table* by *scheme* boundaries."""
    if boundary_table.num_rows() == 0:
        return []
    key_table = plc.Table([table.columns()[key.column_index] for key in scheme.keys])
    split_col = plc.search.lower_bound(
        key_table,
        boundary_table,
        [key.order for key in scheme.keys],
        [key.null_order for key in scheme.keys],
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


def _pack_table(table: plc.Table, stream: Stream, br: BufferResource) -> PackedData:
    """Pack a pylibcudf table as a RapidsMPF PackedData payload."""
    return PackedData.from_cudf_packed_columns(
        pack(table, stream, mr=br.device_mr),
        stream,
        br,
    )


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


def _materialize_packed_pieces(
    pieces: Iterable[PackedData],
    context: Context,
    stream: Stream,
) -> TableChunk | None:
    """Materialize packed pieces into a uniquely-owned chunk."""
    pieces = list(pieces)
    if not pieces:
        return None
    table = unpack_and_concat(pieces, stream=stream, br=context.br())
    if table.num_rows() == 0:
        return None
    return TableChunk.from_pylibcudf_table(
        table,
        stream,
        exclusive_view=True,
        br=context.br(),
    )


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
    metadata.
    """
    _validate_schemes(input_scheme, output_scheme)
    npartitions = output_scheme.num_boundaries + 1
    local_pids = _local_partitions(comm.rank, comm.nranks, npartitions)

    if comm.nranks > 1 and collective_id is None:
        raise ValueError("collective_id is required when comm.nranks > 1.")

    # TODO: Narrow the peer list (avoid all-to-all control messages)
    peers = [rank for rank in range(comm.nranks) if rank != comm.rank]
    exchange = (
        SparseAlltoall(context, comm, collective_id, srcs=peers, dsts=peers)
        if comm.nranks > 1
        else None
    )
    boundary_chunk = output_scheme.get_boundaries(context.br())
    boundary_table = boundary_chunk.table_view()
    local_pieces: dict[int, list[PackedData]] = defaultdict(list)

    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg, br=context.br()).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        if chunk.table_view().num_rows() == 0:
            continue
        with stream_ordered_after(
            context.get_stream_from_pool,
            upstreams=(chunk.stream, boundary_chunk.stream),
        ) as stream:
            table = chunk.table_view()
            splits = _split_points(table, boundary_table, output_scheme, stream)
            for pid, piece in enumerate(
                plc.copying.split(table, splits, stream=stream)
            ):
                if piece.num_rows() == 0:
                    continue
                owner = _contiguous_owner(pid, comm.nranks, npartitions)
                if owner == comm.rank:
                    local_pieces[pid].append(_pack_table(piece, stream, context.br()))
                else:
                    assert exchange is not None
                    exchange.insert(
                        owner,
                        _pack_table(
                            _append_partition_id(piece, pid, stream),
                            stream,
                            context.br(),
                        ),
                    )

    if exchange is not None:
        await exchange.insert_finished(context)

    output_chunks: dict[int, list[TableChunk]] = defaultdict(list)
    for source_rank in range(comm.nranks):
        if source_rank == comm.rank:
            for pid, pieces in local_pieces.items():
                chunk = _materialize_packed_pieces(
                    pieces, context, context.get_stream_from_pool()
                )
                if chunk is not None:
                    output_chunks[pid].append(chunk)
            continue
        assert exchange is not None
        stream = context.get_stream_from_pool()
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
