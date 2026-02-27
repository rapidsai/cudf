# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sort (ShuffleSorted) logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.coll.shuffler import PartitionAssignment
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata, Partitioning
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import Sort
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
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
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


@define_actor()
async def sort_node(
    context: Context,
    ir: Any,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    by: list[str],
    column_order: tuple[Any, ...],
    null_order: tuple[Any, ...],
    column_names: list[str],
    by_dtypes: list[DataType],
    num_partitions: int,
    collective_ids: list[int],
) -> None:
    async with shutdown_on_error(context, ch_in, ch_out):
        metadata_in = await recv_metadata(ch_in, context)
        stream = ir_context.get_cuda_stream()
        by_indices = [column_names.index(b) for b in by]
        my_part_id = context.comm().rank

        chunks_buffer: list[TableChunk] = []
        local_sort_tables: list[plc.Table] = []

        while (msg := await ch_in.recv(context)) is not None:
            chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            chunks_buffer.append(chunk)

        # Build combined from sort columns of *sorted* chunks so boundary
        # candidates are sampled from globally sorted order. Keep the
        # made-available chunks for the insert loop (avoid released chunks).
        chunks_for_insert: list[TableChunk] = []
        for chunk in chunks_buffer:
            chunk = chunk.make_available_and_spill(context.br(), allow_overbooking=True)
            chunks_for_insert.append(chunk)
            tbl = chunk.table_view()
            sort_cols_tbl = plc.Table([tbl.columns()[i] for i in by_indices])
            sorted_tbl = plc.sorting.sort_by_key(
                tbl,
                sort_cols_tbl,
                list(column_order),
                list(null_order),
                stream=stream,
            )
            sort_cols = plc.Table([sorted_tbl.columns()[i] for i in by_indices])
            local_sort_tables.append(sort_cols)

        if not local_sort_tables:
            combined = plc.Table(
                [
                    plc.column_factories.make_empty_column(
                        by_dtypes[i].plc_type, stream=stream
                    )
                    for i in range(len(by))
                ]
            )
        else:
            combined = plc.concatenate.concatenate(local_sort_tables, stream=stream)
        del local_sort_tables

        combined_df = DataFrame.from_table(combined, by, by_dtypes, stream=stream)
        local_candidates_df = _select_local_split_candidates(
            combined_df, by, num_partitions, my_part_id
        )
        candidate_names = list(local_candidates_df.column_names)
        candidate_dtypes = list(local_candidates_df.dtypes)
        candidates_chunk = TableChunk.from_pylibcudf_table(
            local_candidates_df.table, stream, exclusive_view=True
        )

        allgather_id = collective_ids[0]
        shuffle_id = collective_ids[1]
        allgather = AllGatherManager(context, allgather_id)
        allgather.insert(my_part_id, candidates_chunk)
        allgather.insert_finished()
        concat_table = await allgather.extract_concatenated(stream, ordered=True)

        concat_df = DataFrame.from_table(
            concat_table, candidate_names, candidate_dtypes, stream=stream
        )
        sort_boundaries_df = _get_final_sort_boundaries(
            concat_df, list(column_order), list(null_order), num_partitions
        )
        # Copy boundaries table so we own the data (concat_df may be freed).
        n = sort_boundaries_df.table.num_rows()
        init = plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream)
        step = plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=stream)
        gather_map = plc.filling.sequence(n, init, step, stream=stream)
        sort_boundaries_tbl = plc.copying.gather(
            sort_boundaries_df.table,
            gather_map,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            stream=stream,
        )

        shuffle = ShuffleManager(
            context,
            num_partitions,
            (),
            shuffle_id,
            partition_assignment=PartitionAssignment.CONTIGUOUS,
        )
        skip_insert = metadata_in.duplicated and context.comm().rank != 0

        for chunk in chunks_for_insert:
            if skip_insert:
                continue
            chunk = chunk.make_available_and_spill(context.br(), allow_overbooking=True)
            tbl = chunk.table_view()
            sort_cols_tbl = plc.Table([tbl.columns()[i] for i in by_indices])
            sorted_tbl = plc.sorting.sort_by_key(
                tbl,
                sort_cols_tbl,
                list(column_order),
                list(null_order),
                stream=stream,
            )
            sort_cols_tbl = plc.Table([sorted_tbl.columns()[i] for i in by_indices])
            splits = find_sort_splits(
                sort_cols_tbl,
                sort_boundaries_tbl,
                my_part_id,
                list(column_order),
                list(null_order),
                stream=stream,
                chunk_relative=True,
            )
            sorted_chunk = TableChunk.from_pylibcudf_table(
                sorted_tbl, stream, exclusive_view=True
            )
            shuffle.insert_chunk_sorted(sorted_chunk, splits)

        await shuffle.insert_finished()

        output_metadata = ChannelMetadata(
            local_count=max(1, num_partitions // context.comm().nranks),
            partitioning=Partitioning(inter_rank=None, local="inherit"),
        )
        await send_metadata(ch_out, context, output_metadata)

        sort_ir = ir.children[0]  # Sort node for do_evaluate
        column_names_list = list(ir.schema.keys())
        dtypes_list = [ir.schema[n] for n in column_names_list]

        for partition_id in sorted(shuffle.local_partitions()):
            out_table = await shuffle.extract_chunk(partition_id, stream)
            # Partition is built from multiple locally-sorted chunks; sort so
            # output is globally ordered (same as tasks sort).
            if out_table.num_rows() > 0:
                df = DataFrame.from_table(
                    out_table,
                    column_names_list,
                    dtypes_list,
                    stream,
                )
                result_df = sort_ir.do_evaluate(
                    *sort_ir._non_child_args,
                    df,
                    context=ir_context,
                )
                out_table = result_df.table
            await ch_out.send(
                context,
                Message(
                    partition_id,
                    TableChunk.from_pylibcudf_table(
                        out_table, stream, exclusive_view=True
                    ),
                ),
            )

        await ch_out.drain(context)


@generate_ir_sub_network.register(Sort)
def _sort_passthrough(ir: Sort, rec: SubNetGenerator) -> tuple[dict, dict]:
    (child,) = ir.children
    nodes, channels = rec(child)
    nodes[ir] = []
    channels[ir] = channels[child]
    return nodes, channels


@generate_ir_sub_network.register(ShuffleSorted)
def _shuffle_sorted_network(
    ir: ShuffleSorted, rec: SubNetGenerator
) -> tuple[dict, dict]:
    (child,) = ir.children
    nodes, channels = rec(child)
    by = [ne.value.name for ne in ir.by if isinstance(ne.value, Col)]
    if len(by) != len(ir.by):
        raise NotImplementedError("Sorting columns must be column names.")
    column_names = list(ir.schema.keys())
    by_dtypes = [ir.schema[b] for b in by]
    num_partitions = rec.state["partition_info"][ir].count
    collective_ids = list(rec.state["collective_id_map"][ir])
    assert len(collective_ids) == 2

    channels[ir] = ChannelManager(rec.state["context"])
    nodes[ir] = [
        sort_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            by=by,
            column_order=ir.order,
            null_order=ir.null_order,
            column_names=column_names,
            by_dtypes=by_dtypes,
            num_partitions=num_partitions,
            collective_ids=collective_ids,
        )
    ]
    return nodes, channels
