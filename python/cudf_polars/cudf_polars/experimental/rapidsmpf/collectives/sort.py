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

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Col
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    chunk_to_frame,
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

    from cudf_polars.containers import DataType
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


async def _compute_sort_boundaries(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    local_candidates_list: list[DataFrame],
    by: list[str],
    by_dtypes: list[DataType],
    num_partitions: int,
    my_part_id: int,
    column_order: list[Any],
    null_order: list[Any],
    allgather_id: int,
) -> plc.Table:
    """
    Allgather per-chunk split candidates and compute the final sort-boundary table.

    Each rank contributes concatenated local candidates; returns a table of
    boundaries (sort columns plus partition_id and local_row) used by
    find_sort_splits for partitioning sorted chunks.
    """
    stream = ir_context.get_cuda_stream()
    if not local_candidates_list:
        empty_sort_tbl = plc.Table(
            [
                plc.column_factories.make_empty_column(
                    by_dtypes[i].plc_type, stream=stream
                )
                for i in range(len(by))
            ]
        )
        empty_df = DataFrame.from_table(empty_sort_tbl, by, by_dtypes, stream=stream)
        local_candidates_list = [
            _select_local_split_candidates(empty_df, by, num_partitions, my_part_id)
        ]
    combined_candidates_tbl = plc.concatenate.concatenate(
        [c.table for c in local_candidates_list], stream=stream
    )
    candidate_names = list(local_candidates_list[0].column_names)
    candidate_dtypes = list(local_candidates_list[0].dtypes)
    candidates_chunk = TableChunk.from_pylibcudf_table(
        combined_candidates_tbl, stream, exclusive_view=True
    )

    allgather = AllGatherManager(context, comm, allgather_id)
    allgather.insert(my_part_id, candidates_chunk)
    allgather.insert_finished()
    concat_table = await allgather.extract_concatenated(stream, ordered=True)

    concat_df = DataFrame.from_table(
        concat_table, candidate_names, candidate_dtypes, stream=stream
    )
    sort_boundaries_df = _get_final_sort_boundaries(
        concat_df, column_order, null_order, num_partitions
    )
    n = sort_boundaries_df.table.num_rows()
    init = plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream)
    step = plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=stream)
    gather_map = plc.filling.sequence(n, init, step, stream=stream)
    return plc.copying.gather(
        sort_boundaries_df.table,
        gather_map,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )


@define_actor()
async def sort_node(
    context: Context,
    comm: Communicator,
    ir: Any,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    by: list[str],
    column_order: tuple[Any, ...],
    null_order: tuple[Any, ...],
    num_partitions: int,
    collective_ids: list[int],
) -> None:
    """Streaming sort (ShuffleSorted): local sort, boundary allgather, shuffle by splits."""
    async with shutdown_on_error(context, ch_in, ch_out):
        metadata_in = await recv_metadata(ch_in, context)
        output_metadata = ChannelMetadata(
            local_count=max(1, num_partitions // comm.nranks),
            partitioning=Partitioning(inter_rank=None, local="inherit"),
        )
        await send_metadata(ch_out, context, output_metadata)

        by_indices = names_to_indices(tuple(by), ir.schema)
        by_dtypes = [ir.schema[b] for b in by]
        my_part_id = comm.rank

        chunks_buffer: list[TableChunk] = []
        local_candidates_list: list[DataFrame] = []
        while (msg := await ch_in.recv(context)) is not None:
            # Local sort
            df = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                chunk_to_frame(
                    TableChunk.from_message(msg).make_available_and_spill(
                        context.br(), allow_overbooking=True
                    ),
                    ir.children[0],
                ),
                context=ir_context,
            )  # type: ignore[call-arg]
            # Select local split candidates for this chunk
            local_candidates_list.append(
                _select_local_split_candidates(
                    DataFrame.from_table(
                        plc.Table([df.table.columns()[i] for i in by_indices]),
                        by,
                        by_dtypes,
                        stream=df.stream,
                    ),
                    by,
                    num_partitions,
                    my_part_id,
                )
            )
            chunks_buffer.append(
                TableChunk.from_pylibcudf_table(
                    df.table,
                    df.stream,
                    exclusive_view=True,
                )
            )
            del df

        sort_boundaries_tbl = await _compute_sort_boundaries(
            context,
            comm,
            ir_context,
            local_candidates_list,
            by,
            by_dtypes,
            num_partitions,
            my_part_id,
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
        for chunk in chunks_buffer:
            if skip_insert:
                continue
            available_chunk = chunk.make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            tbl = available_chunk.table_view()
            # Chunk is already sorted; only need sort columns for split indices.
            sort_cols_tbl = plc.Table([tbl.columns()[i] for i in by_indices])
            splits = find_sort_splits(
                sort_cols_tbl,
                sort_boundaries_tbl,
                my_part_id,
                list(column_order),
                list(null_order),
                stream=available_chunk.stream,
                chunk_relative=True,
            )
            shuffle.insert_chunk_sorted(available_chunk, splits)

        await shuffle.insert_finished()

        sort_ir = ir.children[0]  # Sort node for do_evaluate
        column_names_list = list(ir.schema.keys())
        dtypes_list = [ir.schema[n] for n in column_names_list]

        for partition_id in sorted(shuffle.local_partitions()):
            stream = ir_context.get_cuda_stream()
            out_table = await shuffle.extract_chunk(partition_id, stream)
            # Partition is built from multiple locally-sorted chunks.
            # Sort concatenated chunk one more tome.
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
        sort_node(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            by=by,
            column_order=ir.order,
            null_order=ir.null_order,
            num_partitions=num_partitions,
            collective_ids=collective_ids,
        )
    ]
    return nodes, channels
