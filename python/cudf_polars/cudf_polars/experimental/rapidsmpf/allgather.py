# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""AllGather logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from rapidsmpf.integrations.cudf.partition import (
    unpack_and_concat as py_unpack_and_concat,
)
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather

from pylibcudf.contiguous_split import pack

if TYPE_CHECKING:
    from types import TracebackType

    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.table_chunk import TableChunk

    import pylibcudf as plc
    from rmm.pylibrmm.stream import Stream


class AllGatherContext:
    """
    AllGather context manager.

    Parameters
    ----------
    context: Context
        The streaming context.
    op_id: int
        Pre-allocated operation ID for this operation.
    """

    def __init__(self, context: Context, op_id: int):
        self.context = context
        self.op_id = op_id
        self._insertion_finished = False

    def __enter__(self) -> AllGatherContext:
        """Enter the AllGatherContext."""
        self.allgather = AllGather(self.context, self.op_id)
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit the AllGatherContext."""
        del self.allgather
        return False

    def insert_chunk(self, chunk: TableChunk) -> None:
        """
        Insert a chunk into the AllGatherContext.

        Parameters
        ----------
        chunk: TableChunk
            The table chunk to insert.
        """
        self.allgather.insert(
            self.context.comm().rank,  # sequence number
            PackedData.from_cudf_packed_columns(
                pack(
                    chunk.table_view(),
                    chunk.stream,
                ),
                chunk.stream,
                self.context.br(),
            ),
        )

    async def extract_concatenated(self, stream: Stream) -> plc.Table:
        """
        Extract the concatenated result.

        Parameters
        ----------
        stream: Stream
            The stream to use for chunk extraction.

        Returns
        -------
        The concatenated AllGather result.
        """
        if not self._insertion_finished:
            self.allgather.insert_finished()
            self._insertion_finished = True

        partition_chunks = await self.allgather.extract_all(self.context, ordered=True)
        return py_unpack_and_concat(
            partitions=partition_chunks,
            stream=stream,
            br=self.context.br(),
        )
