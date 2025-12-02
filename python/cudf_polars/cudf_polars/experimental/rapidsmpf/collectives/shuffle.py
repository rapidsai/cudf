# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack as py_partition_and_pack,
    unpack_and_concat as py_unpack_and_concat,
)
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.expr import Col
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    Metadata,
    empty_table_chunk,
)
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    import pylibcudf as plc
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


class ShuffleManager:
    """
    ShufflerAsync manager.

    Parameters
    ----------
    context: Context
        The streaming context.
    num_partitions: int
        The number of partitions to shuffle into.
    columns_to_hash: tuple[int, ...]
        The columns to hash.
    collective_id: int
        The collective ID.
    """

    def __init__(
        self,
        context: Context,
        num_partitions: int,
        columns_to_hash: tuple[int, ...],
        collective_id: int,
    ):
        self.context = context
        self.num_partitions = num_partitions
        self.columns_to_hash = columns_to_hash
        self.shuffler = ShufflerAsync(
            context,
            collective_id,
            num_partitions,
        )

    def insert_chunk(self, chunk: TableChunk) -> None:
        """
        Insert a chunk into the ShuffleContext.

        Parameters
        ----------
        chunk: TableChunk
            The table chunk to insert.
        """
        # Partition and pack using the Python function
        partitioned_chunks = py_partition_and_pack(
            table=chunk.table_view(),
            columns_to_hash=self.columns_to_hash,
            num_partitions=self.num_partitions,
            stream=chunk.stream,
            br=self.context.br(),
        )

        # Insert into shuffler
        self.shuffler.insert(partitioned_chunks)

    async def insert_finished(self) -> None:
        """Insert finished into the ShuffleManager."""
        await self.shuffler.insert_finished(self.context)

    async def extract_chunk(self, sequence_number: int, stream: Stream) -> plc.Table:
        """
        Extract a chunk from the ShuffleManager.

        Parameters
        ----------
        sequence_number: int
            The sequence number of the chunk to extract.
        stream: Stream
            The stream to use for chunk extraction.

        Returns
        -------
        The extracted table.
        """
        partition_chunks = await self.shuffler.extract_async(
            self.context, sequence_number
        )
        return py_unpack_and_concat(
            partitions=partition_chunks,
            stream=stream,
            br=self.context.br(),
        )


@define_py_node()
async def shuffle_node(
    context: Context,
    ir: Shuffle,
    ir_context: IRExecutionContext,
    ch_in: ChannelPair,
    ch_out: ChannelPair,
    columns_to_hash: tuple[int, ...],
    num_partitions: int,
    collective_id: int,
) -> None:
    """
    Execute a local shuffle pipeline in a single node.

    This node combines partition_and_pack, shuffler, and unpack_and_concat
    into a single Python node using rapidsmpf.shuffler.Shuffler and utilities
    from rapidsmpf.integrations.cudf.partition.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Shuffle IR node.
    ir_context
        The execution context for the IR node.
    ch_in
        Input ChannelPair with metadata and data channels.
    ch_out
        Output ChannelPair with metadata and data channels.
    columns_to_hash
        Tuple of column indices to use for hashing.
    num_partitions
        Number of partitions to shuffle into.
    collective_id
        The collective ID.
    """
    async with shutdown_on_error(
        context, ch_in.metadata, ch_in.data, ch_out.metadata, ch_out.data
    ):
        # Receive and send updated metadata.
        _ = await ch_in.recv_metadata(context)
        column_names = list(ir.schema.keys())
        partitioned_on = tuple(column_names[i] for i in columns_to_hash)
        output_metadata = Metadata(
            max(1, num_partitions // context.comm().nranks),
            partitioned_on=partitioned_on,
        )
        await ch_out.send_metadata(context, output_metadata)

        # Create ShuffleManager instance
        shuffle = ShuffleManager(
            context, num_partitions, columns_to_hash, collective_id
        )

        # Process input chunks
        while (msg := await ch_in.data.recv(context)) is not None:
            # Extract TableChunk from message
            chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )

            # Get the table view and insert into shuffler
            shuffle.insert_chunk(chunk)

        # Insert finished
        await shuffle.insert_finished()

        # Extract shuffled partitions and send them out
        num_partitions_local = 0
        stream = ir_context.get_cuda_stream()
        for partition_id in range(
            # Round-robin partition assignment
            context.comm().rank,
            num_partitions,
            context.comm().nranks,
        ):
            # Create a new TableChunk with the result
            output_chunk = TableChunk.from_pylibcudf_table(
                table=await shuffle.extract_chunk(partition_id, stream),
                stream=stream,
                exclusive_view=True,
            )

            # Send the output chunk
            await ch_out.data.send(context, Message(partition_id, output_chunk))
            num_partitions_local += 1

        # Make sure we send at least one chunk.
        # This can happen during multi-GPU execution.
        # TODO: Investigate and address the underlying issue(s)
        # with skipping this empty-table message.
        if num_partitions_local < 1:
            await ch_out.data.send(
                context,
                Message(
                    num_partitions + 1,
                    empty_table_chunk(ir, context, stream),
                ),
            )

        await ch_out.data.drain(context)


@generate_ir_sub_network.register(Shuffle)
def _(
    ir: Shuffle, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Local shuffle operation.

    # Process children
    (child,) = ir.children
    nodes, channels = rec(child)

    keys: list[Col] = [ne.value for ne in ir.keys if isinstance(ne.value, Col)]
    if len(keys) != len(ir.keys):  # pragma: no cover
        raise NotImplementedError("Shuffle requires simple keys.")
    column_names = list(ir.schema.keys())

    context = rec.state["context"]
    columns_to_hash = tuple(column_names.index(k.name) for k in keys)
    num_partitions = rec.state["partition_info"][ir].count

    # Look up the reserved collective ID for this operation
    collective_id = rec.state["collective_id_map"][ir]

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Complete shuffle node
    nodes[ir] = [
        shuffle_node(
            context,
            ir,
            rec.state["ir_context"],
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            columns_to_hash=columns_to_hash,
            num_partitions=num_partitions,
            collective_id=collective_id,
        )
    ]

    return nodes, channels
