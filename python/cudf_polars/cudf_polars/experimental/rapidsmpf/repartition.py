# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Re-chunking logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import ChannelManager, Metadata
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


@define_py_node()
async def concatenate_node(
    context: Context,
    ir: Repartition,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    *,
    output_count: int,
    collective_id: int,
) -> None:
    """
    Concatenate node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Repartition IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    ch_in
        The input ChannelPair.
    output_count
        The expected number of output chunks.
    collective_id
        Pre-allocated collective ID for this operation.
    """
    async with shutdown_on_error(
        context, ch_in.metadata, ch_in.data, ch_out.metadata, ch_out.data
    ):
        # Receive metadata.
        input_metadata = await ch_in.recv_metadata(context)
        metadata = Metadata(output_count)

        # max_chunks corresponds to the number of chunks we can
        # concatenate together. If None, we must concatenate everything.
        # Since a single-partition operation gets "special treatment",
        # we must make sure `output_count == 1` is always satisfied.
        max_chunks: int | None = None
        if output_count > 1:
            # Make sure max_chunks is at least 2.
            max_chunks = max(2, math.ceil(input_metadata.count / output_count))

        # Check if we need global communication.
        need_global_repartition = (
            # Avoid allgather of already-duplicated data
            not input_metadata.duplicated and output_count == 1
        )

        chunks: list[TableChunk]
        msg: TableChunk | None
        if need_global_repartition:
            # Assume this means "global repartitioning" for now

            # Send metadata.
            metadata.duplicated = True
            await ch_out.send_metadata(context, metadata)

            allgather = AllGatherManager(context, collective_id)
            stream = context.get_stream_from_pool()
            seq_num = 0
            while (msg := await ch_in.data.recv(context)) is not None:
                allgather.insert(seq_num, TableChunk.from_message(msg))
                seq_num += 1
            allgather.insert_finished()
            await ch_out.data.send(
                context,
                Message(
                    0,
                    TableChunk.from_pylibcudf_table(
                        await allgather.extract_concatenated(stream),
                        stream,
                        exclusive_view=True,
                    ),
                ),
            )
        else:
            # Send metadata.
            metadata.duplicated = input_metadata.duplicated
            await ch_out.send_metadata(context, metadata)

            # Local repartitioning
            seq_num = 0
            while True:
                chunks = []
                msg = None

                # Collect chunks up to max_chunks or until end of stream
                while len(chunks) < (max_chunks or float("inf")):
                    msg = await ch_in.data.recv(context)
                    if msg is None:
                        break
                    chunks.append(
                        TableChunk.from_message(msg).make_available_and_spill(
                            context.br(), allow_overbooking=True
                        )
                    )

                # Process collected chunks
                if chunks:
                    df = _concat(
                        *(
                            DataFrame.from_table(
                                chunk.table_view(),
                                list(ir.schema.keys()),
                                list(ir.schema.values()),
                                chunk.stream,
                            )
                            for chunk in chunks
                        ),
                        context=ir_context,
                    )
                    await ch_out.data.send(
                        context,
                        Message(
                            seq_num,
                            TableChunk.from_pylibcudf_table(
                                df.table, df.stream, exclusive_view=True
                            ),
                        ),
                    )
                    seq_num += 1

                # Break if we reached end of stream
                if msg is None:
                    break

        await ch_out.data.drain(context)


@generate_ir_sub_network.register(Repartition)
def _(
    ir: Repartition, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Repartition node.

    partition_info = rec.state["partition_info"]
    if partition_info[ir].count > 1:
        count_output = partition_info[ir].count
        count_input = partition_info[ir.children[0]].count
        if count_input < count_output:
            raise ValueError("Repartitioning to more chunks is not supported.")

    # Process children
    nodes, channels = rec(ir.children[0])

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Look up the reserved shuffle ID for this operation
    collective_id = rec.state["collective_id_map"][ir]

    # Add python node
    nodes[ir] = [
        concatenate_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            output_count=partition_info[ir].count,
            collective_id=collective_id,
        )
    ]
    return nodes, channels
