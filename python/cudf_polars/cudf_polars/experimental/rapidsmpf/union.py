# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Union logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import Union
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import define_actor, shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    empty_table_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    send_metadata,
)
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


@define_actor()
async def union_node(
    context: Context,
    comm: Communicator,
    ir: Union,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    *chs_in: Channel[TableChunk],
    collective_ids: list[int],
) -> None:
    """
    Union node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Union IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    chs_in
        The input Channel[TableChunk]s.
    collective_ids
        Pre-allocated collective IDs for per-child AllGather operations.
    """
    async with shutdown_on_error(
        context, *chs_in, ch_out, trace_ir=ir, ir_context=ir_context
    ):
        # Merge and forward metadata.
        # Union loses partitioning/ordering info since sources may differ.
        # TODO: Warn users that Union does NOT preserve order?
        metadata = await gather_in_task_group(
            *(recv_metadata(ch, context) for ch in chs_in)
        )

        if comm.nranks > 1:
            # Gather each child independently so child order is preserved before
            # redistributing the concatenated result into rank-contiguous slices.
            await send_metadata(
                ch_out,
                context,
                ChannelMetadata(local_count=1),
            )

            stream = context.get_stream_from_pool()
            dfs: list[DataFrame] = []
            collective_id = collective_ids[0]
            for ch_in, meta in zip(chs_in, metadata, strict=True):
                allgather = AllGatherManager(context, comm, collective_id)
                with allgather.inserting() as inserter:
                    seq_num = 0
                    skip_insert = meta.duplicated and comm.rank != 0
                    while (msg := await ch_in.recv(context)) is not None:
                        if not skip_insert:
                            inserter.insert(
                                seq_num,
                                TableChunk.from_message(msg, br=context.br()),
                            )
                            seq_num += 1
                        del msg

                table = await allgather.extract_concatenated(
                    stream, ir_context=ir_context
                )
                if table.num_columns() == 0 and len(ir.schema) > 0:
                    chunk = empty_table_chunk(ir, context, stream)
                    table = chunk.table_view()
                    stream = chunk.stream
                dfs.append(
                    DataFrame.from_table(
                        table,
                        list(ir.schema.keys()),
                        list(ir.schema.values()),
                        stream,
                    )
                )

            df = _concat(*dfs, context=ir_context)
            rows_per_rank = max(1, (df.num_rows + comm.nranks - 1) // comm.nranks)
            offset = rows_per_rank * comm.rank
            length = max(0, min(rows_per_rank, df.num_rows - offset))
            df = df.slice((offset, length))
            await ch_out.send(
                context,
                Message(
                    0,
                    TableChunk.from_pylibcudf_table(
                        df.table,
                        df.stream,
                        exclusive_view=True,
                        br=context.br(),
                    ),
                ),
            )
            del df
            await ch_out.drain(context)
            return

        # Chunk counts on the wire are uniform across ranks, so report the
        # full sum.
        total_local_count = sum(meta.local_count for meta in metadata)
        duplicated = all(meta.duplicated for meta in metadata)
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(
                local_count=total_local_count,
                duplicated=duplicated,
            ),
        )

        seq_num_offset = 0
        for ch_in in chs_in:
            num_ch_chunks = 0
            while (msg := await ch_in.recv(context)) is not None:
                out_chunk = TableChunk.from_message(
                    msg, br=context.br()
                ).make_available_and_spill(context.br(), allow_overbooking=True)
                await ch_out.send(
                    context,
                    Message(
                        msg.sequence_number + seq_num_offset,
                        out_chunk,
                    ),
                )
                num_ch_chunks += 1
            seq_num_offset += num_ch_chunks

        await ch_out.drain(context)


@generate_ir_sub_network.register(Union)
def _(
    ir: Union, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Union operation.
    # Pass-through all child chunks in channel order.

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Add simple python node
    nodes[ir] = [
        union_node(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            *[channels[c].reserve_output_slot() for c in ir.children],
            collective_ids=rec.state["collective_id_map"][ir],
        )
    ]
    return nodes, channels
