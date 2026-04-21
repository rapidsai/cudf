# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack as py_partition_and_pack,
    split_and_pack as py_split_and_pack,
    unpack_and_concat as py_unpack_and_concat,
)
from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    _is_already_partitioned,
    chunk_to_frame,
    recv_metadata,
    send_metadata,
)
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


class ShuffleManager:
    """
    ShufflerAsync manager.

    Parameters
    ----------
    context: Context
        The streaming context.
    comm: Communicator
        The communicator.
    num_partitions: int
        The number of partitions to shuffle into.
    collective_id: int
        The collective ID.
    partition_assignment: PartitionAssignment, optional
        How to assign partition IDs to ranks: ROUND_ROBIN (default) or
        CONTIGUOUS. Use CONTIGUOUS for sort so each rank gets adjacent
        partition IDs and concatenation order matches global order.
    """

    class Inserter:
        """
        Context manager for the insert phase of a shuffle operation.

        Obtained via :meth:`ShuffleManager.inserting`. On exit, signals
        the end of insertion to all ranks by calling ``insert_finished()``.

        Parameters
        ----------
        manager: ShuffleManager
            The shuffle manager to insert into.
        """

        def __init__(self, manager: ShuffleManager):
            self._manager = manager

        def insert_hash(
            self, chunk: TableChunk, columns_to_hash: tuple[int, ...]
        ) -> None:
            """Partition chunk by hash and insert into the shuffler."""
            self._manager.shuffler.insert(
                py_partition_and_pack(
                    table=chunk.table_view(),
                    columns_to_hash=columns_to_hash,
                    num_partitions=self._manager.num_partitions,
                    stream=chunk.stream,
                    br=self._manager.context.br(),
                )
            )

        def insert_hash_with_keys(
            self, chunk: TableChunk, key_table: plc.Table
        ) -> None:
            """
            Partition chunk by hash using a separate key table and insert.

            Uses ``hash_partition(input, key_table, ...)`` to support
            non-``Col`` (e.g. expression-derived) shuffle keys.
            """
            partitioned_table, offsets = plc.partitioning.hash_partition(
                chunk.table_view(),
                key_table,
                self._manager.num_partitions,
                stream=chunk.stream,
            )
            self._manager.shuffler.insert(
                py_split_and_pack(
                    table=partitioned_table,
                    splits=list(offsets[1:-1]),
                    stream=chunk.stream,
                    br=self._manager.context.br(),
                )
            )

        def insert_split(self, chunk: TableChunk, splits: list[int]) -> None:
            """Split chunk at the given indices and insert into the shuffler."""
            self._manager.shuffler.insert(
                py_split_and_pack(
                    table=chunk.table_view(),
                    splits=splits,
                    stream=chunk.stream,
                    br=self._manager.context.br(),
                )
            )

        async def __aenter__(self) -> ShuffleManager.Inserter:
            """Enter the context manager."""
            return self

        async def __aexit__(self, *args: object) -> None:
            """Exit the context manager, calling ``insert_finished()``."""
            await self._manager.shuffler.insert_finished(self._manager.context)

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        num_partitions: int,
        collective_id: int,
        *,
        partition_assignment: PartitionAssignment = PartitionAssignment.ROUND_ROBIN,
    ):
        self.context = context
        self.num_partitions = num_partitions
        self.shuffler = ShufflerAsync(
            context,
            comm,
            collective_id,
            num_partitions,
            partition_assignment=partition_assignment,
        )

    def inserting(self) -> ShuffleManager.Inserter:
        """Return a context manager for the insert phase."""
        return ShuffleManager.Inserter(self)

    def local_partitions(self) -> list[int]:
        """Get the local partition IDs for this rank."""
        return self.shuffler.local_partitions()

    def extract_chunk(self, sequence_number: int, stream: Stream) -> plc.Table:
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

        Raises
        ------
        KeyError
            If the requested sequence number has already been extracted.
        """
        partition_chunks = self.shuffler.extract(sequence_number)
        return py_unpack_and_concat(
            partitions=partition_chunks,
            stream=stream,
            br=self.context.br(),
        )


async def _global_shuffle(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    num_partitions: int,
    collective_id: int,
    keys_to_hash: tuple[NamedExpr, ...],
    child_ir: IR,
) -> None:
    """
    Global shuffle implementation.

    Parameters
    ----------
    context
        The streaming context.
    comm
        The communicator.
    ir_context
        The execution context for the IR node.
    ch_out
        Output Channel[TableChunk] with metadata and data channels.
    ch_in
        Input Channel[TableChunk] with metadata and data channels.
    num_partitions
        Number of partitions to shuffle into.
    collective_id
        The collective ID.
    keys_to_hash
        Tuple of ``NamedExpr`` objects to evaluate at runtime as the
        hash keys. Key columns are computed by evaluating these
        expressions on each incoming chunk.
    child_ir
        The child IR node, used to wrap incoming chunks as DataFrames
        for expression evaluation.
    """
    # For Col-only keys derive column indices so we can check whether
    # the data is already correctly partitioned and skip the shuffle.
    key_values = [ne.value for ne in keys_to_hash]
    col_keys = [k for k in key_values if isinstance(k, Col)]
    col_indices: tuple[int, ...] | None = None
    if len(col_keys) == len(key_values):
        schema_keys = list(child_ir.schema.keys())
        col_indices = tuple(schema_keys.index(k.name) for k in col_keys)

    metadata_in = await recv_metadata(ch_in, context)

    if col_indices is not None and _is_already_partitioned(
        metadata_in, col_indices, num_partitions, comm.nranks
    ):
        await send_metadata(ch_out, context, metadata_in)
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
        await ch_out.drain(context)
        return

    # For Col-only keys describe the output partitioning so downstream
    # operations can skip redundant shuffles. For expression-based keys
    # we can't express the partitioning in terms of column indices.
    output_metadata = ChannelMetadata(
        local_count=max(1, num_partitions // comm.nranks),
        partitioning=Partitioning(
            inter_rank=HashScheme(col_indices, num_partitions),
            local="inherit",
        )
        if col_indices is not None
        else None,
    )
    await send_metadata(ch_out, context, output_metadata)

    # When input is duplicated, only rank 0 should contribute data.
    # Other ranks still participate in the shuffle protocol.
    skip_insert = metadata_in.duplicated and comm.rank != 0

    shuffle = ShuffleManager(context, comm, num_partitions, collective_id)
    async with shuffle.inserting() as inserter:
        while (msg := await ch_in.recv(context)) is not None:
            if not skip_insert:
                chunk = TableChunk.from_message(
                    msg, br=context.br()
                ).make_available_and_spill(context.br(), allow_overbooking=True)
                df = chunk_to_frame(chunk, child_ir)
                key_table = DataFrame(
                    [expr.evaluate(df) for expr in keys_to_hash],
                    stream=df.stream,
                ).table
                inserter.insert_hash_with_keys(chunk, key_table)

    for partition_id in shuffle.shuffler.local_partitions():
        stream = ir_context.get_cuda_stream()
        await ch_out.send(
            context,
            Message(
                partition_id,
                TableChunk.from_pylibcudf_table(
                    table=shuffle.extract_chunk(partition_id, stream),
                    stream=stream,
                    exclusive_view=True,
                    br=context.br(),
                ),
            ),
        )

    await ch_out.drain(context)


@define_actor()
async def shuffle_actor(
    context: Context,
    comm: Communicator,
    ir: Shuffle,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    num_partitions: int,
    collective_id: int,
    keys_to_hash: tuple[NamedExpr, ...],
    child_ir: IR,
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
    comm
        The communicator.
    ir
        The Shuffle IR node.
    ir_context
        The execution context for the IR node.
    ch_in
        Input Channel[TableChunk] with metadata and data channels.
    ch_out
        Output Channel[TableChunk] with metadata and data channels.
    num_partitions
        Number of partitions to shuffle into.
    collective_id
        The collective ID.
    keys_to_hash
        Tuple of ``NamedExpr`` objects to evaluate at runtime as the
        hash keys.
    child_ir
        The child IR node, used to wrap incoming chunks as DataFrames
        for expression evaluation.
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ):
        await _global_shuffle(
            context,
            comm,
            ir_context,
            ch_out,
            ch_in,
            num_partitions,
            collective_id,
            keys_to_hash,
            child_ir,
        )


@generate_ir_sub_network.register(Shuffle)
def _(
    ir: Shuffle, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Local shuffle operation.

    # Process children
    (child,) = ir.children
    nodes, channels = rec(child)

    key_values = [ne.value for ne in ir.keys]

    # Non-pointwise expressions (e.g. aggregations) cannot be evaluated
    # chunk-by-chunk and are therefore unsupported as shuffle keys.
    if not all(expr.is_pointwise for expr in traversal(key_values)):  # pragma: no cover
        raise NotImplementedError("Shuffle requires pointwise key expressions.")

    context = rec.state["context"]
    num_partitions = rec.state["partition_info"][ir].count
    collective_id = rec.state["collective_id_map"][ir][0]

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Complete shuffle node
    nodes[ir] = [
        shuffle_actor(
            context,
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            num_partitions=num_partitions,
            collective_id=collective_id,
            keys_to_hash=ir.keys,
            child_ir=child,
        )
    ]

    return nodes, channels
