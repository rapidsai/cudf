# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.communicator.single import new_communicator
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack as py_partition_and_pack,
    unpack_and_concat as py_unpack_and_concat,
)
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.expr import Col
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import ChannelManager
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from types import TracebackType

    from rapidsmpf.streaming.core.context import Context

    import pylibcudf as plc
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


# TODO: This implementation only supports a single GPU for now.
#       Multi-GPU support will require a distinct GlobalShuffle
#       context manager, and updated _shuffle_id_vacancy logic.


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


class LocalShuffle:
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
    """

    def __init__(
        self,
        context: Context,
        num_partitions: int,
        columns_to_hash: tuple[int, ...],
    ):
        self.context = context
        self.br = context.br()
        self.num_partitions = num_partitions
        self.columns_to_hash = columns_to_hash
        self._insertion_finished = False

    def __enter__(self) -> LocalShuffle:
        """Enter the local shuffle instance context manager."""
        self.op_id = _get_new_shuffle_id()
        statistics = self.context.statistics()
        comm = new_communicator(Options(get_environment_variables()))
        progress_thread = ProgressThread(comm, statistics)
        self.shuffler = Shuffler(
            comm=comm,
            progress_thread=progress_thread,
            op_id=self.op_id,
            total_num_partitions=self.num_partitions,
            br=self.br,
            statistics=statistics,
        )
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit the local shuffle instance context manager."""
        self.shuffler.shutdown()
        _release_shuffle_id(self.op_id)
        return False

    def insert_chunk(self, chunk: TableChunk) -> None:
        """
        Insert a chunk into the local shuffle instance.

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
            br=self.br,
        )

        # Insert into shuffler
        self.shuffler.insert_chunks(partitioned_chunks)

    def extract_chunk(self, sequence_number: int, stream: Stream) -> plc.Table:
        """
        Extract a chunk from the local shuffle instance.

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
            self.shuffler.insert_finished(list(range(self.num_partitions)))
            self._insertion_finished = True

        self.shuffler.wait_on(sequence_number)
        partition_chunks = self.shuffler.extract(sequence_number)
        return py_unpack_and_concat(
            partitions=partition_chunks,
            stream=stream,
            br=self.br,
        )


@define_py_node()
async def local_shuffle_node(
    context: Context,
    ir: Shuffle,
    ir_context: IRExecutionContext,
    ch_in: ChannelPair,
    ch_out: ChannelPair,
    columns_to_hash: tuple[int, ...],
    num_partitions: int,
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
    """
    async with shutdown_on_error(
        context, ch_in.metadata, ch_in.data, ch_out.metadata, ch_out.data
    ):
        # Create LocalShuffle context manager to handle shuffler lifecycle
        # TODO: Use ir_context to get the stream (not available yet)
        with LocalShuffle(context, num_partitions, columns_to_hash) as local_shuffle:
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
                local_shuffle.insert_chunk(chunk)

            # Extract shuffled partitions and send them out
            # LocalShuffle.extract_chunk handles insert_finished, wait, extract, and unpack
            stream = ir_context.get_cuda_stream()
            for partition_id in range(num_partitions):
                # Create a new TableChunk with the result
                output_chunk = TableChunk.from_pylibcudf_table(
                    table=local_shuffle.extract_chunk(partition_id, stream),
                    stream=stream,
                    exclusive_view=True,
                )

                # Send the output chunk
                await ch_out.data.send(context, Message(partition_id, output_chunk))

        await ch_out.data.drain(context)


@generate_ir_sub_network.register(Shuffle)
def _(ir: Shuffle, rec: SubNetGenerator) -> tuple[list[Any], dict[IR, ChannelManager]]:
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

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Complete shuffle pipeline in a single node
    # LocalShuffle context manager handles shuffle ID lifecycle internally
    nodes.append(
        local_shuffle_node(
            context,
            ir,
            rec.state["ir_context"],
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            columns_to_hash=columns_to_hash,
            num_partitions=num_partitions,
        )
    )

    return nodes, channels
