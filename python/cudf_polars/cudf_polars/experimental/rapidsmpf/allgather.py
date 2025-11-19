# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""AllGather logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from rapidsmpf.allgather import AllGather
from rapidsmpf.integrations.cudf.partition import (
    unpack_and_concat as py_unpack_and_concat,
)
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.progress_thread import ProgressThread

from pylibcudf.contiguous_split import pack

from cudf_polars.experimental.rapidsmpf.shuffle import (
    _get_new_shuffle_id,
    _release_shuffle_id,
)

if TYPE_CHECKING:
    from types import TracebackType

    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.table_chunk import TableChunk

    import pylibcudf as plc
    from rmm.pylibrmm.stream import Stream


class GlobalAllGather:
    """
    global AllGather context manager.

    Parameters
    ----------
    context: Context
        The streaming context.
    """

    def __init__(self, context: Context):
        self.context = context
        self._insertion_finished = False

    def __enter__(self) -> GlobalAllGather:
        """Enter the GlobalAllGather context manager."""
        self.op_id = _get_new_shuffle_id()
        statistics = self.context.statistics()
        progress_thread = ProgressThread(self.context.comm(), statistics)
        self.allgather = AllGather(
            comm=self.context.comm(),
            progress_thread=progress_thread,
            op_id=self.op_id,
            br=self.context.br(),
            statistics=statistics,
        )
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit the GlobalAllGather context manager."""
        del self.allgather
        _release_shuffle_id(self.op_id)
        return False

    def insert_chunk(self, chunk: TableChunk) -> None:
        """
        Insert a chunk into the GlobalAllGather instance.

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

    def mark_insertion_finished(self) -> None:
        """Mark insertion as finished."""
        if not self._insertion_finished:
            self.allgather.insert_finished()
            self._insertion_finished = True

    def extract_concatenated(self, stream: Stream) -> plc.Table:
        """
        Extract the concatenated AllGather result.

        Parameters
        ----------
        stream: Stream
            The stream to use for chunk extraction.

        Returns
        -------
        The concatenated AllGather result.
        """
        self.mark_insertion_finished()
        partition_chunks = self.allgather.wait_and_extract()
        return py_unpack_and_concat(
            partitions=partition_chunks,
            stream=stream,
            br=self.context.br(),
        )
