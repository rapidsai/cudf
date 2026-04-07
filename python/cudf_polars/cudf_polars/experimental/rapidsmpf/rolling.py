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

import polars as pl

import pylibcudf as plc
from pylibcudf.contiguous_split import pack

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.ir import Rolling
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    allgather_reduce,
    concat_batch,
    empty_table_chunk,
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
    """Convert a raw polars ordinal (nanoseconds for datetime, plain int for INT64) to native column units."""
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


def _pack_table(table: plc.Table, stream: Stream, br: Any) -> PackedData | None:
    """Pack a plc.Table into PackedData; return None if the table is empty."""
    if table.num_rows() == 0:
        return None
    return PackedData.from_cudf_packed_columns(
        pack(table, stream=stream, mr=br.device_mr),
        stream,
        br,
    )


def _compute_halos(
    local_table: plc.Table,
    index_col_idx: int,
    index_dtype: plc.DataType,
    lookback: int,
    lookahead: int,
    stream: Stream,
    br: Any,
) -> tuple[PackedData | None, PackedData | None]:
    """
    Compute initial halo data to send to neighboring ranks.

    Parameters
    ----------
    local_table
        The locally sorted data table.
    index_col_idx
        Column index of the rolling index column.
    index_dtype
        pylibcudf data type of the index column.
    lookback
        Lookback distance in native column units (>= 0).
        Rows with index >= (max_idx - lookback) are sent rightward.
    lookahead
        Lookahead distance in native column units (>= 0).
        Rows with index <= (min_idx + lookahead) are sent leftward.
    stream
        CUDA stream.
    br
        Memory buffer resource.

    Returns
    -------
    (send_left, send_right)
        Packed halo data to send to the left and right neighbors.
    """
    if local_table.num_rows() == 0:
        return None, None

    # Get the index column; cast to INT64 for uniform arithmetic
    index_col = local_table.columns()[index_col_idx]
    if index_dtype.id() == plc.TypeId.INT64:
        index_col_i64 = index_col
    else:
        index_col_i64 = plc.unary.cast(
            index_col, plc.DataType(plc.TypeId.INT64), stream=stream
        )

    # Min/max of index as Python ints (D→H transfer of 2 scalars)
    i64 = plc.DataType(plc.TypeId.INT64)
    min_scalar = plc.reduce.reduce(
        index_col_i64, plc.aggregation.min(), i64, stream=stream
    )
    max_scalar = plc.reduce.reduce(
        index_col_i64, plc.aggregation.max(), i64, stream=stream
    )
    minmax_df = DataFrame.from_table(
        plc.Table(
            [
                plc.Column.from_scalar(min_scalar, 1, stream=stream),
                plc.Column.from_scalar(max_scalar, 1, stream=stream),
            ]
        ),
        ["min_idx", "max_idx"],
        [DataType(pl.Int64()), DataType(pl.Int64())],
        stream=stream,
    ).to_polars()
    min_idx: int = minmax_df["min_idx"][0]
    max_idx: int = minmax_df["max_idx"][0]

    bool8 = plc.DataType(plc.TypeId.BOOL8)

    # send_right: rows with index >= (max_idx - lookback) for rank k+1's lookback
    send_right: PackedData | None = None
    if lookback > 0:
        thr_r = plc.Scalar.from_py(max_idx - lookback, i64, stream=stream)
        mask_r = plc.binaryop.binary_operation(
            index_col_i64,
            thr_r,
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            bool8,
            stream=stream,
        )
        send_right = _pack_table(
            plc.stream_compaction.apply_boolean_mask(
                local_table, mask_r, stream=stream
            ),
            stream,
            br,
        )

    # send_left: rows with index <= (min_idx + lookahead) for rank k-1's lookahead
    send_left: PackedData | None = None
    if lookahead > 0:
        thr_l = plc.Scalar.from_py(min_idx + lookahead, i64, stream=stream)
        mask_l = plc.binaryop.binary_operation(
            index_col_i64,
            thr_l,
            plc.binaryop.BinaryOperator.LESS_EQUAL,
            bool8,
            stream=stream,
        )
        send_left = _pack_table(
            plc.stream_compaction.apply_boolean_mask(
                local_table, mask_l, stream=stream
            ),
            stream,
            br,
        )

    return send_left, send_right


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
    1. Buffer all locally sorted input chunks.
    2. For nranks == 1: evaluate Rolling directly and return.
    3. Compute initial halos (boundary rows to send to neighbors).
    4. HaloExchange loop: relay halos until all ranks converge (allgather_reduce).
    5. Concat accumulated halos + local data + right halos.
    6. Evaluate Rolling on the combined table (no zlice).
    7. Strip halo rows; apply ir.zlice; send result.
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        await recv_metadata(ch_in, context)
        await send_metadata(
            ch_out, context, ChannelMetadata(local_count=1, partitioning=None)
        )

        # ---- Buffer all local chunks ----------------------------------------
        child_schema = ir.children[0].schema
        chunks: list[TableChunk] = []
        while (msg := await ch_in.recv(context)) is not None:
            chunks.append(
                TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
            )

        if chunks:
            local_chunk = await concat_batch(chunks, context, child_schema, ir_context)
        else:
            local_chunk = empty_table_chunk(
                ir.children[0], context, ir_context.get_cuda_stream()
            )
        del chunks

        local_chunk = local_chunk.make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        local_table = local_chunk.table_view()
        n_local_rows = local_table.num_rows()
        col_names = list(child_schema.keys())
        col_dtypes = list(child_schema.values())

        # ---- Fast path for nranks == 1 ---------------------------------------
        if comm.nranks == 1:
            non_child_args_no_zlice = ir._non_child_args[:-1] + (None,)
            local_df = DataFrame.from_table(
                local_table, col_names, col_dtypes, stream=local_chunk.stream
            )
            result_df = await asyncio.to_thread(
                Rolling.do_evaluate,
                *non_child_args_no_zlice,
                local_df,
                context=ir_context,
            )
            if ir.zlice is not None:
                result_df = result_df.slice(ir.zlice)
            result_chunk = TableChunk.from_pylibcudf_table(
                result_df.table, result_df.stream, exclusive_view=True
            )
            if tracer is not None:
                tracer.add_chunk(table=result_chunk.table_view())
            await ch_out.send(context, Message(0, result_chunk))
            await ch_out.drain(context)
            return

        # ---- Compute window distances in native column units -----------------
        type_id = ir.index_dtype.id()
        preceding_native = _ordinal_to_native(type_id, ir.preceding_ordinal)
        following_native = _ordinal_to_native(type_id, ir.following_ordinal)
        # lookback: how far BACK the window looks (from each row's index)
        # e.g. preceding_ordinal=-3 → lookback=3 for "period=3i, offset=-3i"
        lookback = max(0, -preceding_native)
        # lookahead: how far FORWARD the window looks
        lookahead = max(0, preceding_native + following_native)

        # ---- Find the index column position in child schema ------------------
        index_col_idx = col_names.index(ir.index.name)

        # ---- Compute initial halos -------------------------------------------
        stream = ir_context.get_cuda_stream()
        br = context.br()
        send_left, send_right = _compute_halos(
            local_table, index_col_idx, ir.index_dtype, lookback, lookahead, stream, br
        )

        # ---- HaloExchange loop -----------------------------------------------
        halo_exchange_id = collective_ids.pop()
        allreduce_id = collective_ids.pop()
        he = HaloExchange(context, comm, halo_exchange_id)

        # Accumulate halos: left halos are prepended (farther data first),
        # right halos are appended (closer data first).
        halo_left_pds: list[PackedData] = []
        halo_right_pds: list[PackedData] = []

        while True:
            from_left, from_right = await he.exchange(send_left, send_right)

            if from_left is not None:
                halo_left_pds.insert(0, from_left)  # prepend: farther data first
            if from_right is not None:
                halo_right_pds.append(from_right)

            # Terminate when all ranks stop receiving new halo data
            my_done = (from_left is None) and (from_right is None)
            (total_done,) = await allgather_reduce(
                context, comm, allreduce_id, int(my_done)
            )
            if total_done == comm.nranks:
                break

            # Relay received halos to the opposite side for multi-hop coverage
            send_right = from_left
            send_left = from_right

        # ---- Assemble combined table and evaluate rolling --------------------
        halo_left_tables: list[plc.Table] = [
            await asyncio.to_thread(
                unpack_and_concat, partitions=[pd], stream=stream, br=br
            )
            for pd in halo_left_pds
        ]
        halo_right_tables: list[plc.Table] = [
            await asyncio.to_thread(
                unpack_and_concat, partitions=[pd], stream=stream, br=br
            )
            for pd in halo_right_pds
        ]

        n_halo_left_rows = sum(t.num_rows() for t in halo_left_tables)

        all_tables = [*halo_left_tables, local_table, *halo_right_tables]
        combined_table = (
            plc.concatenate.concatenate(all_tables, stream=stream)
            if len(all_tables) > 1
            else all_tables[0]
        )
        combined_df = DataFrame.from_table(
            combined_table, col_names, col_dtypes, stream=stream
        )

        # Evaluate rolling without zlice (applied after stripping halos)
        non_child_args_no_zlice = ir._non_child_args[:-1] + (None,)
        result_df = await asyncio.to_thread(
            Rolling.do_evaluate,
            *non_child_args_no_zlice,
            combined_df,
            context=ir_context,
        )

        # Strip halo rows and apply zlice
        local_result_df = result_df.slice((n_halo_left_rows, n_local_rows))
        if ir.zlice is not None:
            local_result_df = local_result_df.slice(ir.zlice)

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
