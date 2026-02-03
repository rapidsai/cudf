# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
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
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.expr import Col
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    recv_metadata,
    send_metadata,
)
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    import pylibcudf as plc
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


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


def _is_already_partitioned(
    metadata: ChannelMetadata,
    columns_to_hash: tuple[int, ...],
    num_partitions: int,
) -> bool:
    """Check if data is already partitioned on the required keys."""
    if metadata.partitioning is None:
        return False

    # Check that inter_rank is a HashScheme (not None or "inherit")
    inter_rank = metadata.partitioning.inter_rank
    if not isinstance(inter_rank, HashScheme):
        return False

    # Check that local partitioning is inherit
    if metadata.partitioning.local != "inherit":
        return False

    # Check for exact match: same columns and same modulus
    return (
        inter_rank.column_indices == columns_to_hash
        and inter_rank.modulus == num_partitions
    )


@define_py_node()
async def shuffle_node(
    context: Context,
    ir: Shuffle,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    columns_to_hash: tuple[int, ...],
    num_partitions: int,
    collective_id: int,
) -> None:
    """
    Execute a global shuffle pipeline within a single node.

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
        Input Channel[TableChunk] with metadata and data channels.
    ch_out
        Output Channel[TableChunk] with metadata and data channels.
    columns_to_hash
        Tuple of column indices to use for hashing.
    num_partitions
        Number of partitions to shuffle into.
    collective_id
        The collective ID.
    """
    async with shutdown_on_error(context, ch_in, ch_out):
        # Receive input metadata
        metadata_in = await recv_metadata(ch_in, context)

        # Check if we can skip the shuffle (already partitioned correctly)
        if _is_already_partitioned(metadata_in, columns_to_hash, num_partitions):
            # Forward metadata and data unchanged
            await send_metadata(ch_out, context, metadata_in)
            while (msg := await ch_in.recv(context)) is not None:
                await ch_out.send(context, msg)
            await ch_out.drain(context)
            return

        # Normal shuffle path
        output_metadata = ChannelMetadata(
            local_count=max(1, num_partitions // context.comm().nranks),
            partitioning=Partitioning(
                inter_rank=HashScheme(columns_to_hash, num_partitions),
                local="inherit",
            ),
        )
        await send_metadata(ch_out, context, output_metadata)

        # Create ShuffleManager instance
        shuffle = ShuffleManager(
            context, num_partitions, columns_to_hash, collective_id
        )

        # When input is duplicated, only rank 0 should contribute data.
        # Other ranks still participate in the shuffle protocol.
        skip_insert = metadata_in.duplicated and context.comm().rank != 0

        # Process input chunks
        while (msg := await ch_in.recv(context)) is not None:
            if not skip_insert:
                # Extract TableChunk from message and insert into shuffler
                shuffle.insert_chunk(
                    TableChunk.from_message(msg).make_available_and_spill(
                        context.br(), allow_overbooking=True
                    )
                )
            del msg

        # Insert finished
        await shuffle.insert_finished()

        # Extract shuffled partitions and send them out
        stream = ir_context.get_cuda_stream()
        for partition_id in range(
            # Round-robin partition assignment
            context.comm().rank,
            num_partitions,
            context.comm().nranks,
        ):
            # Extract and send the output chunk
            await ch_out.send(
                context,
                Message(
                    partition_id,
                    TableChunk.from_pylibcudf_table(
                        table=await shuffle.extract_chunk(partition_id, stream),
                        stream=stream,
                        exclusive_view=True,
                    ),
                ),
            )

        await ch_out.drain(context)


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
    collective_id = rec.state["collective_id_map"][ir][0]

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
