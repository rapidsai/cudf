# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""AllGather logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather

from pylibcudf.contiguous_split import pack

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.table_chunk import TableChunk

    import pylibcudf as plc
    from rmm.pylibrmm.stream import Stream


class AllGatherManager:
    """
    AllGather manager.

    Parameters
    ----------
    context: Context
        The streaming context.
    comm: Communicator
        The communicator.
    op_id: int
        Pre-allocated operation ID for this operation.
    """

    class Inserter:
        """
        Context manager for the insert phase of an AllGather operation.

        Obtained via :meth:`AllGatherManager.inserting`. On exit, signals
        the end of insertion to all ranks by calling ``insert_finished()``.

        Parameters
        ----------
        manager: AllGatherManager
            The AllGather manager to insert into.
        """

        def __init__(self, manager: AllGatherManager):
            self._manager = manager

        def insert(self, sequence_number: int, chunk: TableChunk) -> None:
            """
            Insert a chunk into the AllGather.

            Parameters
            ----------
            sequence_number: int
                The sequence number of the chunk to insert.
            chunk: TableChunk
                The table chunk to insert. Need not be GPU-resident; if spilled,
                it will be made available internally.
            """
            chunk = chunk.make_available_and_spill(
                self._manager.context.br(), allow_overbooking=True
            )
            self._manager.allgather.insert(
                sequence_number,
                # TODO: Avoid unnecessary copies.
                # See https://github.com/rapidsai/rapidsmpf/issues/933
                PackedData.from_cudf_packed_columns(
                    pack(
                        chunk.table_view(),
                        chunk.stream,
                        mr=self._manager.context.br().device_mr,
                    ),
                    chunk.stream,
                    self._manager.context.br(),
                ),
            )
            del chunk

        def __enter__(self) -> AllGatherManager.Inserter:
            """Enter the context manager."""
            return self

        def __exit__(self, *args: object) -> None:
            """Exit the context manager, calling ``insert_finished()``."""
            self._manager.allgather.insert_finished()

    def __init__(self, context: Context, comm: Communicator, op_id: int):
        self.context = context
        self.allgather = AllGather(self.context, comm, op_id)

    def inserting(self) -> AllGatherManager.Inserter:
        """Return a context manager for the insert phase."""
        return AllGatherManager.Inserter(self)

    async def extract_concatenated(
        self, stream: Stream, *, ordered: bool = True
    ) -> plc.Table:
        """
        Extract the concatenated result.

        Parameters
        ----------
        stream: Stream
            The stream to use for chunk extraction.
        ordered: bool
            Whether to extract the data in ordered or unordered fashion.

        Returns
        -------
        The concatenated AllGather result.
        """
        return await asyncio.to_thread(
            unpack_and_concat,
            partitions=await self.allgather.extract_all(self.context, ordered=ordered),
            stream=stream,
            br=self.context.br(),
        )
