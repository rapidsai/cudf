# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack as py_partition_and_pack,
    unpack_and_concat as py_unpack_and_concat,
)
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import ChannelManager, empty_table_chunk
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from types import TracebackType

    from rapidsmpf.streaming.core.context import Context

    import pylibcudf as plc
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


# Set of available shuffle IDs
_shuffle_id_vacancy: set[int] = set(range(Shuffler.max_concurrent_shuffles))
_shuffle_id_vacancy_lock: threading.Lock = threading.Lock()


def _get_new_shuffle_id() -> int:
    with _shuffle_id_vacancy_lock:
        if not _shuffle_id_vacancy:
            raise ValueError(
                f"Cannot shuffle more than {Shuffler.max_concurrent_shuffles} "
                "times in a single query."
            )

        return _shuffle_id_vacancy.pop()


def _release_shuffle_id(op_id: int) -> None:
    """Release a shuffle ID back to the vacancy set."""
    with _shuffle_id_vacancy_lock:
        _shuffle_id_vacancy.add(op_id)


class ReserveOpIDs:
    """
    Context manager to reserve shuffle IDs for pipeline execution.

    Parameters
    ----------
    ir : IR
        The root IR node of the pipeline.

    Notes
    -----
    This context manager:
    1. Identifies all Shuffle nodes in the IR
    2. Reserves shuffle IDs from the vacancy pool
    3. Creates a mapping from IR nodes to their reserved IDs
    4. Releases all IDs back to the pool on __exit__
    """

    def __init__(self, ir: IR):
        # Collect all Shuffle nodes.
        # NOTE: We will also need to collect Repartition,
        # and Join nodes to support multi-GPU execution.
        self.shuffle_nodes: list[IR] = [
            node for node in traversal([ir]) if isinstance(node, Shuffle)
        ]
        self.shuffle_id_map: dict[IR, int] = {}

    def __enter__(self) -> dict[IR, int]:
        """
        Reserve shuffle IDs and return the mapping.

        Returns
        -------
        shuffle_id_map : dict[IR, int]
            Mapping from IR nodes to their reserved shuffle IDs.
        """
        # Reserve IDs and map nodes directly to their IDs
        for node in self.shuffle_nodes:
            self.shuffle_id_map[node] = _get_new_shuffle_id()

        return self.shuffle_id_map

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Release all reserved shuffle IDs back to the vacancy pool."""
        for op_id in self.shuffle_id_map.values():
            _release_shuffle_id(op_id)
        return False


class ShuffleContext:
    """
    Local shuffle instance context manager.

    Parameters
    ----------
    context: Context
        The streaming context.
    num_partitions: int
        The number of partitions to shuffle into.
    columns_to_hash: tuple[int, ...]
        The columns to hash.
    op_id: int
        The shuffle operation ID.
    """

    def __init__(
        self,
        context: Context,
        num_partitions: int,
        columns_to_hash: tuple[int, ...],
        op_id: int,
    ):
        self.context = context
        self.num_partitions = num_partitions
        self.columns_to_hash = columns_to_hash
        self.op_id = op_id
        self._insertion_finished = False

    def __enter__(self) -> ShuffleContext:
        """Enter the local shuffle instance context manager."""
        self.shuffler = ShufflerAsync(
            self.context,
            self.op_id,
            self.num_partitions,
        )
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit the ShuffleContext manager."""
        del self.shuffler
        return False

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

    async def extract_chunk(self, sequence_number: int, stream: Stream) -> plc.Table:
        """
        Extract a chunk from the ShuffleContext.

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
        if not self._insertion_finished:
            await self.shuffler.insert_finished(self.context)
            self._insertion_finished = True

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
    op_id: int,
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
    op_id
        The shuffle operation ID.
    """
    async with shutdown_on_error(
        context, ch_in.metadata, ch_in.data, ch_out.metadata, ch_out.data
    ):
        # Create ShuffleContext context manager to handle shuffler lifecycle
        with ShuffleContext(context, num_partitions, columns_to_hash, op_id) as shuffle:
            # Process input chunks
            while True:
                msg = await ch_in.data.recv(context)
                if msg is None:
                    break

                # Extract TableChunk from message
                chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )

                # Get the table view and insert into shuffler
                shuffle.insert_chunk(chunk)

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
            if num_partitions_local < 1:
                await ch_out.data.send(
                    context,
                    Message(
                        num_partitions + 1,
                        empty_table_chunk(ir, context),
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

    # Look up the reserved shuffle ID for this operation
    op_id = rec.state["shuffle_id_map"][ir]

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
            op_id=op_id,
        )
    ]

    return nodes, channels
