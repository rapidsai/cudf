# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sort (ShuffleSorted) logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.coll.shuffler import PartitionAssignment
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata, Partitioning
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.dsl.ir import Empty, Sort
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    chunk_to_frame,
    concat_batch,
    empty_table_chunk,
    names_to_indices,
    recv_metadata,
    send_metadata,
)
from cudf_polars.experimental.sort import (
    ShuffleSorted,
    _get_final_sort_boundaries,
    _select_local_split_candidates,
    find_sort_splits,
)

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.typing import Schema


def _boundary_schema(by: list[str], by_dtypes: list[DataType]) -> Schema:
    """Schema of boundaries table."""
    name_gen = unique_names(by)
    part_id_dtype = DataType(pl.UInt32())
    return dict(
        zip(
            [*by, next(name_gen), next(name_gen)],
            [*by_dtypes, part_id_dtype, part_id_dtype],
            strict=True,
        )
    )


async def _compute_sort_boundaries(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    local_candidates_list: list[TableChunk],
    by: list[str],
    by_dtypes: list[DataType],
    num_partitions: int,
    column_order: list[Any],
    null_order: list[Any],
    allgather_id: int,
) -> plc.Table:
    """Compute global sort boundaries."""
    boundary_ir = Empty(_boundary_schema(by, by_dtypes))
    local_boundaries_df = _get_final_sort_boundaries(
        chunk_to_frame(
            await concat_batch(
                local_candidates_list,
                context,
                boundary_ir.schema,
                ir_context,
            )
            if local_candidates_list
            else empty_table_chunk(
                boundary_ir,
                context,
                ir_context.get_cuda_stream(),
            ),
            boundary_ir,
        ),
        column_order,
        null_order,
        num_partitions,
    )
    stream = local_boundaries_df.stream

    if comm.nranks > 1:
        allgather = AllGatherManager(context, comm, allgather_id)
        chunk = TableChunk.from_pylibcudf_table(
            local_boundaries_df.table,
            stream,
            exclusive_view=True,
        )
        allgather.insert(comm.rank, chunk)
        allgather.insert_finished()
        concat_table = await allgather.extract_concatenated(stream, ordered=True)
        boundaries_df = _get_final_sort_boundaries(
            DataFrame.from_table(
                concat_table,
                list(boundary_ir.schema.keys()),
                list(boundary_ir.schema.values()),
                stream=stream,
            ),
            column_order,
            null_order,
            num_partitions,
        )
    else:
        boundaries_df = local_boundaries_df

    if boundaries_df.table.num_rows() > 0:
        return plc.copying.gather(
            boundaries_df.table,
            plc.filling.sequence(
                boundaries_df.table.num_rows(),
                plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream),
                plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=stream),
                stream=stream,
            ),
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            stream=stream,
        )
    else:
        return boundaries_df.table


@define_actor()
async def sort_actor(
    context: Context,
    comm: Communicator,
    ir: ShuffleSorted,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    by: list[str],
    num_partitions: int,
    collective_ids: list[int],
) -> None:
    """Streaming sort actor."""
    async with shutdown_on_error(context, ch_in, ch_out):
        metadata_in = await recv_metadata(ch_in, context)
        output_metadata = ChannelMetadata(
            local_count=max(1, num_partitions // comm.nranks),
            partitioning=Partitioning(inter_rank=None, local="inherit"),
        )
        await send_metadata(ch_out, context, output_metadata)

        column_order = list(ir.order)
        null_order = list(ir.null_order)
        by_indices = names_to_indices(tuple(by), ir.schema)
        by_dtypes = [ir.schema[b] for b in by]
        sort_ir = ir.children[0]
        assert isinstance(sort_ir, Sort), "ShuffleSorted must have a Sort child."
        seq_id_name = next(unique_names(ir.schema.keys())) if sort_ir.stable else None

        message_ids: list[int] = []
        chunks_buffer = context.spillable_messages()
        local_candidates_list: list[TableChunk] = []
        local_row_offset = 0

        while (msg := await ch_in.recv(context)) is not None:
            seq_num = msg.sequence_number
            # Incoming chunks are locally sorted by the upstream Sort
            df = await asyncio.to_thread(
                chunk_to_frame,
                TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                ),
                ir.children[0],
            )
            local_candidates_list.append(
                TableChunk.from_pylibcudf_table(
                    _select_local_split_candidates(
                        df.select(by),
                        by,
                        num_partitions,
                        seq_num,
                    ).table,
                    df.stream,
                    exclusive_view=True,
                )
            )
            if sort_ir.stable:
                nrows = df.table.num_rows()
                # High bits = rank
                # Low bits = local row index.
                start = (comm.rank * (1 << 48)) + local_row_offset
                seq_id_col = plc.filling.sequence(
                    nrows,
                    plc.Scalar.from_py(
                        start, plc.DataType(plc.TypeId.UINT64), stream=df.stream
                    ),
                    plc.Scalar.from_py(
                        1, plc.DataType(plc.TypeId.UINT64), stream=df.stream
                    ),
                    stream=df.stream,
                )
                local_row_offset += nrows
                tbl = plc.Table([*df.table.columns(), seq_id_col])
            else:
                tbl = df.table
            chunk = TableChunk.from_pylibcudf_table(tbl, df.stream, exclusive_view=True)
            mid = chunks_buffer.insert(Message(seq_num, chunk))
            message_ids.append(mid)
            del df

        sort_boundaries_tbl = await _compute_sort_boundaries(
            context,
            comm,
            ir_context,
            local_candidates_list,
            by,
            by_dtypes,
            num_partitions,
            list(column_order),
            list(null_order),
            collective_ids.pop(),
        )

        shuffle = ShuffleManager(
            context,
            comm,
            num_partitions,
            (),
            collective_ids.pop(),
            partition_assignment=PartitionAssignment.CONTIGUOUS,
        )
        skip_insert = metadata_in.duplicated and comm.rank != 0

        # Insert sorted chunks into the shuffler
        for mid in message_ids:
            msg = chunks_buffer.extract(mid=mid)
            if skip_insert:
                continue
            seq_num = int(msg.sequence_number)
            available_chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            tbl = available_chunk.table_view()
            sort_cols_tbl = plc.Table([tbl.columns()[i] for i in by_indices])
            splits = find_sort_splits(
                sort_cols_tbl,
                sort_boundaries_tbl,
                seq_num,
                list(column_order),
                list(null_order),
                stream=available_chunk.stream,
                chunk_relative=True,
            )
            shuffle.insert_chunk(available_chunk, splits=splits)

        await shuffle.insert_finished()

        if sort_ir.stable:
            assert seq_id_name is not None
            sort_ir = Sort(
                sort_ir.schema | {seq_id_name: DataType(pl.UInt64())},
                (
                    *sort_ir.by,
                    NamedExpr(seq_id_name, Col(DataType(pl.UInt64()), seq_id_name)),
                ),
                (*sort_ir.order, plc.types.Order.ASCENDING),
                (*sort_ir.null_order, plc.types.NullOrder.AFTER),
                sort_ir.stable,
                sort_ir.zlice,
                sort_ir.children[0],
            )

        for partition_id in shuffle.local_partitions():
            stream = ir_context.get_cuda_stream()
            out_table = await shuffle.extract_chunk(partition_id, stream)
            df = DataFrame.from_table(
                out_table,
                list(sort_ir.schema.keys()),
                list(sort_ir.schema.values()),
                stream,
            )
            if out_table.num_rows() > 0:
                df = sort_ir.do_evaluate(
                    *sort_ir._non_child_args,
                    df,
                    context=ir_context,
                )
                out_table = df.table
            if sort_ir.stable:
                # Dropt the stable-sort seq_id column
                out_table = plc.Table(out_table.columns()[:-1])
            await ch_out.send(
                context,
                Message(
                    partition_id,
                    TableChunk.from_pylibcudf_table(
                        out_table, df.stream, exclusive_view=True
                    ),
                ),
            )
            del df

        await ch_out.drain(context)


@generate_ir_sub_network.register(ShuffleSorted)
def _shuffle_sorted_network(
    ir: ShuffleSorted, rec: SubNetGenerator
) -> tuple[dict, dict]:
    (child,) = ir.children
    nodes, channels = rec(child)
    by = [ne.value.name for ne in ir.by if isinstance(ne.value, Col)]
    if len(by) != len(ir.by):
        raise NotImplementedError("Sorting columns must be column names.")
    num_partitions = rec.state["partition_info"][ir].count
    collective_ids = list(rec.state["collective_id_map"][ir])
    assert len(collective_ids) == 2

    channels[ir] = ChannelManager(rec.state["context"])
    nodes[ir] = [
        sort_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            by=by,
            num_partitions=num_partitions,
            collective_ids=collective_ids,
        )
    ]
    return nodes, channels
