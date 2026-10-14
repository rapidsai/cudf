# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-overlap exchange for streaming operators."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall

from cudf_polars.streaming.actor_graph.utils import RandomAccessChunkStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.message import Message


@dataclass(frozen=True)
class StoredChunk:
    """Storage ID and row span for one buffered chunk."""

    mid: int
    sequence_number: int
    row_start: int
    num_rows: int

    @property
    def row_stop(self) -> int:
        """Return the first row after this chunk."""
        return self.row_start + self.num_rows


class BufferedInput:
    """Spillable input supporting ordered replay and row-slice access."""

    def __init__(self, context: Context) -> None:
        self._context = context
        self._store = RandomAccessChunkStore(context)
        self._chunks: list[StoredChunk] = []
        self._row_offset = 0

    @property
    def chunks(self) -> tuple[StoredChunk, ...]:
        """Return buffered chunk metadata in input order."""
        return tuple(self._chunks)

    def insert(self, msg: Message, *, row_start: int, num_rows: int) -> None:
        """Store one input message and its rank-local row span."""
        sequence_number = msg.sequence_number
        mid = self._store.insert(msg)
        self._chunks.append(StoredChunk(mid, sequence_number, row_start, num_rows))

    def set_row_offset(self, row_offset: int) -> None:
        """Set the global row offset for this rank's input."""
        self._row_offset = row_offset

    def clear(self) -> None:
        """Release all buffered messages."""
        self._store.clear()
        self._chunks.clear()

    def replay(self) -> Iterator[Message]:
        """Yield and remove buffered messages in input order."""
        for chunk in self._chunks:
            yield self._store.extract(chunk.mid)
        self._chunks.clear()

    @asynccontextmanager
    async def borrow(self, chunk: StoredChunk) -> AsyncIterator[TableChunk]:
        """Temporarily make one buffered chunk available."""
        async with self._store.borrow(chunk.mid) as table_chunk:
            yield table_chunk

    async def slices(self, start: int, stop: int) -> AsyncIterator[TableChunk]:
        """Yield views covering a global half-open row range."""
        expected = start
        for chunk in self._chunks:
            chunk_start = chunk.row_start + self._row_offset
            chunk_stop = chunk.row_stop + self._row_offset
            if chunk_stop <= start:
                continue
            if chunk_start >= stop:
                break
            piece_start = max(start, chunk_start)
            piece_stop = min(stop, chunk_stop)
            if piece_start < piece_stop:
                if piece_start != expected:
                    raise RuntimeError("Buffered input does not cover requested rows")
                async with self._store.borrow(chunk.mid) as table_chunk:
                    local_start = piece_start - chunk_start
                    local_stop = piece_stop - chunk_start
                    if local_start == 0 and local_stop == chunk.num_rows:
                        table = plc.Table(table_chunk.table_view().columns())
                    else:
                        (table,) = plc.copying.slice(
                            table_chunk.table_view(),
                            [local_start, local_stop],
                            stream=table_chunk.stream,
                        )
                    yield TableChunk.from_pylibcudf_table(
                        table,
                        table_chunk.stream,
                        exclusive_view=False,
                        br=self._context.br(),
                    )
                expected = piece_stop
        if expected != stop:
            raise RuntimeError("Buffered input does not cover requested rows")


@dataclass(frozen=True)
class GhostSend:
    """A global row span to send to another rank."""

    destination: int
    start: int
    stop: int


class RowExchange:
    """Exchange planned row slices between ranks."""

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        source: BufferedInput,
        sends: Sequence[GhostSend],
        sources: Sequence[int],
        destinations: Sequence[int],
        *,
        collective_id: int,
    ) -> None:
        self._context = context
        self._source = source
        self._sends = tuple(sends)
        self.sources = tuple(sorted(set(sources)))
        destinations = tuple(sorted(set(destinations)))
        self._exchange = (
            SparseAlltoall(
                context,
                comm,
                collective_id,
                srcs=self.sources,
                dsts=destinations,
            )
            if self.sources or destinations
            else None
        )

    async def prepare(self) -> None:
        """Insert all outgoing slices and finish the exchange."""
        if self._exchange is None:
            return
        try:
            for send in sorted(
                self._sends, key=lambda item: (item.destination, item.start)
            ):
                async for chunk in self._source.slices(send.start, send.stop):
                    available, extra = await make_table_chunks_available_or_wait(
                        self._context,
                        chunk,
                        reserve_extra=chunk.into_packed_data_cost(),
                        net_memory_delta=0,
                    )
                    self._exchange.insert(
                        send.destination, available.into_packed_data(extra)
                    )
        finally:
            await self._exchange.insert_finished(self._context)

    def receive(self, source: int) -> Iterator[TableChunk]:
        """Yield ghost chunks received from one source rank."""
        assert self._exchange is not None
        for packed in self._exchange.extract(source):
            yield TableChunk.from_packed_data(packed, self._context.br())
