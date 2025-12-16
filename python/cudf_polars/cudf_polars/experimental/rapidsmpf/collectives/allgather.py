# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
    op_id: int
        Pre-allocated operation ID for this operation.
    """

    def __init__(self, context: Context, op_id: int):
        self.context = context
        self.allgather = AllGather(self.context, op_id)

    def insert(self, sequence_number: int, chunk: TableChunk) -> None:
        """
        Insert a chunk into the AllGatherContext.

        Parameters
        ----------
        sequence_number: int
            The sequence number of the chunk to insert.
        chunk: TableChunk
            The table chunk to insert.
        """
        self.allgather.insert(
            sequence_number,
            PackedData.from_cudf_packed_columns(
                pack(
                    chunk.table_view(),
                    chunk.stream,
                ),
                chunk.stream,
                self.context.br(),
            ),
        )

    def insert_finished(self) -> None:
        """Insert finished into the AllGatherManager."""
        self.allgather.insert_finished()

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
        partition_chunks = await self.allgather.extract_all(
            self.context, ordered=ordered
        )
        return await asyncio.to_thread(
            unpack_and_concat,
            partitions=partition_chunks,
            stream=stream,
            br=self.context.br(),
        )
