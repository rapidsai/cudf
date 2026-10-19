# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-overlap planning and exchange for streaming operators."""

from __future__ import annotations

import operator
import struct
from bisect import bisect_left
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.utils.windows import duration_to_scalar
from cudf_polars.streaming.actor_graph.utils import ChunkStore, names_to_indices
from cudf_polars.utils.cuda_stream import join_cuda_streams

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Sequence
    from typing import Never

    from pylibcudf.utils import CudaStreamLike
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.memory_resource import DeviceMemoryResource
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import Rolling


type RowRange = tuple[int, int]
type GhostSend = tuple[int, int, int]


def _unsupported_policy(policy: object) -> Never:
    raise TypeError(f"Unsupported overlap policy: {type(policy).__name__}")


type _BoundSearch = Callable[
    [
        plc.Table,
        plc.Table,
        list[plc.types.Order],
        list[plc.types.NullOrder],
        CudaStreamLike | None,
        DeviceMemoryResource | None,
    ],
    plc.Column,
]


class RangeValidation(IntEnum):
    """Validation status for one rank's range index."""

    VALID = 0
    NULL = 1
    UNSORTED = 2
    OUT_OF_BOUNDS = 3


@dataclass
class BufferedChunk:
    """An available input chunk with its global row span."""

    chunk: TableChunk
    row_start: int
    num_rows: int

    @property
    def row_stop(self) -> int:
        """Return the first row after this chunk."""
        return self.row_start + self.num_rows


@dataclass
class StagedChunk[BufferedChunkT: BufferedChunk]:
    """A prepared chunk marked as locally owned or ghost data."""

    chunk: BufferedChunkT
    owned: bool


@dataclass(frozen=True)
class RangeValues:
    """Host-side endpoints for one nonempty range."""

    first: int
    last: int
    lower: int
    upper: int


type RangeRequest = tuple[int, RangeValues]


@dataclass
class RangeOverlap:
    """Range-window metadata for one buffered input chunk."""

    index_column: plc.Column
    lower_bound_column: plc.Column
    upper_bound_column: plc.Column
    values: RangeValues | None


@dataclass
class RangeBufferedChunk(BufferedChunk):
    """A buffered chunk with range-window overlap metadata."""

    overlap: RangeOverlap


class _OverlapPolicy[BufferedChunkT: BufferedChunk]:
    """Base policy for finding the rows needed by a rolling cursor."""

    def observe(self, chunk: BufferedChunkT) -> None:
        pass

    def close(self) -> None:
        pass

    def validate_cursor(self, cursor: BufferedChunkT) -> None:
        pass

    def evict_history(
        self,
        history: list[BufferedChunkT],
        cursor: BufferedChunkT,
    ) -> list[BufferedChunkT]:
        raise NotImplementedError

    def has_complete_future(
        self,
        latest: BufferedChunkT,
        current: BufferedChunkT,
    ) -> bool:
        raise NotImplementedError

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunkT,
    ) -> BufferedChunkT | None:
        """Return the cursor rows that future cursors may need."""
        del context
        return cursor

    async def overlap_bounds(
        self,
        context: Context,
        cursor: BufferedChunkT,
        chunks: Sequence[BufferedChunkT],
    ) -> tuple[int, int]:
        raise NotImplementedError

    async def prepare_chunk(
        self,
        context: Context,
        chunk: TableChunk,
        *,
        row_offset: int,
        host_values: RangeValues | None = None,
    ) -> BufferedChunkT:
        raise NotImplementedError


@dataclass
class RangeOverlapPolicy(_OverlapPolicy[RangeBufferedChunk]):
    """
    Actor-local policy and execution state for range rolling windows.

    The mutable fields track ordering across chunks and the streams using the
    policy's device scalars. Each rolling actor creates and closes one instance.
    """

    lower: plc.Scalar
    upper: plc.Scalar
    index: int
    index_dtype: plc.DataType
    find_start: _BoundSearch
    find_end: _BoundSearch
    start_closed: bool
    end_closed: bool
    stream: Stream
    index_name: str
    observed_streams: set[Stream] = field(default_factory=set, init=False, repr=False)
    previous_index_value: int | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_ir(cls, ir: Rolling, stream: Stream) -> RangeOverlapPolicy:
        """Create actor-local range-bound state for a rolling node."""
        (index,) = names_to_indices([ir.index.name], ir.children[0].schema)
        side = ir.closed_window
        start_closed = side in ("both", "left")
        end_closed = side in ("both", "right")
        dtype = ir.index_dtype
        return cls(
            # Note: not using windows_to_offsets because that flips the sign of
            # preceding_ordinal.
            duration_to_scalar(dtype, ir.preceding_ordinal, stream=stream),
            duration_to_scalar(
                dtype,
                ir.preceding_ordinal + ir.following_ordinal,
                stream=stream,
            ),
            index,
            dtype,
            plc.search.lower_bound if start_closed else plc.search.upper_bound,
            plc.search.upper_bound if end_closed else plc.search.lower_bound,
            start_closed,
            end_closed,
            stream,
            ir.index.name,
        )

    def observe(self, chunk: RangeBufferedChunk) -> None:
        """Track streams that own range-bound columns."""
        self.observed_streams.add(chunk.chunk.stream)

    def close(self) -> None:
        """Keep offset scalars alive until all observed work is ordered."""
        if self.observed_streams:
            join_cuda_streams(
                downstreams=(self.stream,),
                upstreams=tuple(self.observed_streams),
            )

    @staticmethod
    def _index_with_offset(
        index: plc.Column,
        row: int,
        offset: plc.Scalar,
        dtype: plc.DataType,
        stream: Stream,
        br: BufferResource,
    ) -> plc.Column:
        """Return ``index[row] + offset`` as a single-row device column."""
        (endpoint,) = plc.copying.slice(index, [row, row + 1], stream=stream)
        if endpoint.type() != dtype:
            endpoint = plc.unary.cast(endpoint, dtype, stream=stream)
        return plc.binaryop.binary_operation(
            endpoint,
            offset,
            plc.binaryop.BinaryOperator.ADD,
            dtype,
            stream=stream,
            mr=br.device_mr,
        )

    @staticmethod
    def _chrono_storage_dtype(dtype: plc.DataType) -> plc.DataType:
        """Return the integer storage type for a chrono dtype."""
        if dtype.id() in (plc.TypeId.TIMESTAMP_DAYS, plc.TypeId.DURATION_DAYS):
            return plc.DataType(plc.TypeId.INT32)
        return plc.DataType(plc.TypeId.INT64)

    @classmethod
    def _host_ordering_value(
        cls,
        value: plc.Column,
        *,
        stream: Stream,
        br: BufferResource,
    ) -> int:
        """Copy a single-row ordering value to host."""
        if plc.traits.is_chrono(value.type()):
            value = plc.unary.bit_cast(
                value,
                cls._chrono_storage_dtype(value.type()),
                stream=stream,
                mr=br.device_mr,
            )
        result = value.to_scalar(stream=stream).to_py(stream=stream)
        assert isinstance(result, int)
        return result

    @classmethod
    def _host_ordering_value_at(
        cls,
        column: plc.Column,
        row: int,
        *,
        stream: Stream,
        br: BufferResource,
    ) -> int:
        """Copy one ordering value from a device column to host."""
        (value,) = plc.copying.slice(column, [row, row + 1], stream=stream)
        return cls._host_ordering_value(value, stream=stream, br=br)

    @staticmethod
    async def _insertion_point(
        context: Context,
        index_column: plc.Column,
        needle: plc.Column,
        find: _BoundSearch,
        dtype: plc.DataType,
        stream: Stream,
    ) -> int:
        """Find a needle's insertion point in one available index column."""
        needs_cast = index_column.type() != dtype
        memory_scope = nullcontext()
        if needs_cast:
            reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
                index_column.size() * 8, net_memory_delta=0
            )
            memory_scope = opaque_memory_usage(reservation)
        with memory_scope:
            if needs_cast:
                index_column = plc.unary.cast(index_column, dtype, stream=stream)
            value = (
                find(
                    plc.Table([index_column]),
                    plc.Table([needle]),
                    [plc.types.Order.ASCENDING],
                    [plc.types.NullOrder.AFTER],
                    stream,
                    context.br().device_mr,
                )
                .to_scalar(stream=stream)
                .to_py(stream=stream)
            )
        assert isinstance(value, int)
        return value

    async def insertion_point(
        self,
        context: Context,
        index_column: plc.Column,
        value: int,
        find: _BoundSearch,
        stream: Stream,
    ) -> int:
        """Find a host value's insertion point in an index column."""
        scalar = plc.Scalar.from_py(
            value,
            self._chrono_storage_dtype(self.index_dtype)
            if plc.traits.is_chrono(self.index_dtype)
            else self.index_dtype,
            stream=stream,
        )
        needle = plc.Column.from_scalar(scalar, 1, stream=stream)
        if plc.traits.is_chrono(self.index_dtype):
            needle = plc.unary.bit_cast(
                needle,
                self.index_dtype,
                stream=stream,
                mr=context.br().device_mr,
            )
        return await self._insertion_point(
            context,
            index_column,
            needle,
            find,
            self.index_dtype,
            stream,
        )

    @staticmethod
    async def _global_insertion_row(
        chunks: Sequence[RangeBufferedChunk],
        needle: plc.Column,
        needle_value: int,
        find: _BoundSearch,
        *,
        upper_bound: bool,
        dtype: plc.DataType,
        context: Context,
        needle_stream: Stream,
    ) -> int:
        """Return the globally indexed insertion row of a needle in some chunks."""
        assert len(chunks) > 0
        for chunk in chunks:
            if chunk.num_rows == 0:
                continue
            assert chunk.overlap.values is not None
            if (
                chunk.overlap.values.last <= needle_value
                if upper_bound
                else chunk.overlap.values.last < needle_value
            ):
                continue
            stream = chunk.chunk.stream
            join_cuda_streams(downstreams=[stream], upstreams=[needle_stream])
            insertion_point = await RangeOverlapPolicy._insertion_point(
                context,
                chunk.overlap.index_column,
                needle,
                find,
                dtype,
                stream,
            )
            if insertion_point < chunk.num_rows:
                return chunk.row_start + insertion_point
        # Needle is later than all the chunks we know about.
        return chunks[-1].row_stop

    @staticmethod
    def index_validation(index_column: plc.Column, stream: Stream) -> RangeValidation:
        """Return the validation status for one index column."""
        if index_column.null_count() != 0:
            return RangeValidation.NULL
        if index_column.size() > 1 and not plc.sorting.is_sorted(
            plc.Table([index_column]),
            [plc.types.Order.ASCENDING],
            [plc.types.NullOrder.BEFORE],
            stream=stream,
        ):
            return RangeValidation.UNSORTED
        return RangeValidation.VALID

    def validate_index_column(self, index_column: plc.Column, stream: Stream) -> None:
        """Validate chunk-local ordering before using search primitives."""
        self.raise_validation(self.index_validation(index_column, stream))

    def raise_validation(self, validation: RangeValidation) -> None:
        """Raise the corresponding range-index validation error."""
        if validation is RangeValidation.NULL:
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling may not contain nulls"
            )
        if validation is RangeValidation.UNSORTED:
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling is not sorted, "
                "please sort first"
            )

    def validate_cursor(self, cursor: RangeBufferedChunk) -> None:
        """Check that the index remains sorted across chunk boundaries."""
        if cursor.num_rows == 0:
            return
        assert cursor.overlap.values is not None
        if (
            self.previous_index_value is not None
            and cursor.overlap.values.first < self.previous_index_value
        ):
            self.raise_validation(RangeValidation.UNSORTED)
        self.previous_index_value = cursor.overlap.values.last

    async def prepare_chunk(
        self,
        context: Context,
        chunk: TableChunk,
        *,
        row_offset: int,
        host_values: RangeValues | None = None,
        validate_index: bool = True,
    ) -> RangeBufferedChunk:
        """Extract range metadata from a table chunk."""
        nrows, _ = chunk.shape
        chunk, extra = await make_table_chunks_available_or_wait(
            context,
            chunk,
            reserve_extra=32,
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            index_column = chunk.table_view().columns()[self.index]
            if host_values is None and validate_index:
                self.validate_index_column(index_column, chunk.stream)
            if nrows == 0:
                overlap = RangeOverlap(index_column, index_column, index_column, None)
            else:
                join_cuda_streams(downstreams=(chunk.stream,), upstreams=(self.stream,))
                lower_bound = self._index_with_offset(
                    index_column,
                    0,
                    self.lower,
                    self.index_dtype,
                    chunk.stream,
                    context.br(),
                )
                upper_bound = self._index_with_offset(
                    index_column,
                    nrows - 1,
                    self.upper,
                    self.index_dtype,
                    chunk.stream,
                    context.br(),
                )
                if host_values is None and (
                    validate_index or index_column.null_count() == 0
                ):
                    host_values = RangeValues(
                        self._host_ordering_value_at(
                            index_column, 0, stream=chunk.stream, br=context.br()
                        ),
                        self._host_ordering_value_at(
                            index_column,
                            nrows - 1,
                            stream=chunk.stream,
                            br=context.br(),
                        ),
                        self._host_ordering_value(
                            lower_bound, stream=chunk.stream, br=context.br()
                        ),
                        self._host_ordering_value(
                            upper_bound, stream=chunk.stream, br=context.br()
                        ),
                    )
                overlap = RangeOverlap(
                    index_column,
                    lower_bound,
                    upper_bound,
                    host_values,
                )
        return RangeBufferedChunk(chunk, row_offset, nrows, overlap)

    def evict_history(
        self,
        history: list[RangeBufferedChunk],
        cursor: RangeBufferedChunk,
    ) -> list[RangeBufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        assert cursor.overlap.values is not None
        keep = operator.ge if self.start_closed else operator.gt
        return [
            chunk
            for chunk in history
            if chunk.overlap.values is not None
            and keep(chunk.overlap.values.last, cursor.overlap.values.lower)
        ]

    def has_complete_future(
        self,
        latest: RangeBufferedChunk,
        current: RangeBufferedChunk,
    ) -> bool:
        """Return whether latest contains current's upper insertion point."""
        if latest.overlap.values is None:
            return False
        assert current.overlap.values is not None
        complete = operator.gt if self.end_closed else operator.ge
        return complete(latest.overlap.values.last, current.overlap.values.upper)

    async def overlap_bounds(
        self,
        context: Context,
        cursor: RangeBufferedChunk,
        chunks: Sequence[RangeBufferedChunk],
    ) -> tuple[int, int]:
        """Return the rows needed to evaluate a range cursor."""
        assert cursor.overlap.values is not None
        ghost_start = await self._global_insertion_row(
            chunks,
            cursor.overlap.lower_bound_column,
            cursor.overlap.values.lower,
            self.find_start,
            upper_bound=not self.start_closed,
            dtype=self.index_dtype,
            context=context,
            needle_stream=cursor.chunk.stream,
        )
        ghost_stop = await self._global_insertion_row(
            chunks,
            cursor.overlap.upper_bound_column,
            cursor.overlap.values.upper,
            self.find_end,
            upper_bound=self.end_closed,
            dtype=self.index_dtype,
            context=context,
            needle_stream=cursor.chunk.stream,
        )
        # We must extract at least the whole of the current cursor chunk.
        ghost_start = min(cursor.row_start, ghost_start)
        ghost_stop = max(cursor.row_stop, ghost_stop)
        return ghost_start, ghost_stop


@dataclass
class RowCountOverlapPolicy(_OverlapPolicy[BufferedChunk]):
    """Overlap policy for fixed-size rolling expressions."""

    preceding: int
    following: int

    async def prepare_chunk(
        self,
        context: Context,
        chunk: TableChunk,
        *,
        row_offset: int,
        host_values: RangeValues | None = None,
    ) -> BufferedChunk:
        """Attach row-span metadata to an available chunk."""
        del host_values
        nrows, _ = chunk.shape
        chunk, _ = await make_table_chunks_available_or_wait(
            context, chunk, reserve_extra=0, net_memory_delta=0
        )
        return BufferedChunk(chunk, row_offset, nrows)

    def evict_history(
        self,
        history: list[BufferedChunk],
        cursor: BufferedChunk,
    ) -> list[BufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        ghost_start = max(0, cursor.row_start - self.preceding)
        return [chunk for chunk in history if chunk.row_stop > ghost_start]

    def has_complete_future(
        self,
        latest: BufferedChunk,
        current: BufferedChunk,
    ) -> bool:
        """Return whether latest contains enough leading rows."""
        return latest.row_stop >= current.row_stop + self.following

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunk,
    ) -> BufferedChunk | None:
        """Keep only emitted rows that a future fixed-size window can use."""
        if self.preceding == 0 or cursor.num_rows == 0:
            return None
        start = max(cursor.row_start, cursor.row_stop - self.preceding)
        if start == cursor.row_start:
            return cursor

        stream = cursor.chunk.stream
        (table,) = plc.copying.slice(
            cursor.chunk.table_view(),
            [start - cursor.row_start, cursor.num_rows],
            stream=stream,
        )
        reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
            cursor.chunk.data_alloc_size(), net_memory_delta=0
        )
        with opaque_memory_usage(reservation):
            table = table.copy(stream=stream, mr=context.br().device_mr)
        return BufferedChunk(
            TableChunk.from_pylibcudf_table(
                table,
                stream,
                exclusive_view=True,
                br=context.br(),
            ),
            start,
            cursor.row_stop - start,
        )

    async def overlap_bounds(
        self,
        context: Context,
        cursor: BufferedChunk,
        chunks: Sequence[BufferedChunk],
    ) -> tuple[int, int]:
        """Return the rows needed to evaluate a fixed-size cursor."""
        del context
        ghost_start = max(0, cursor.row_start - self.preceding)
        ghost_stop = min(cursor.row_stop + self.following, chunks[-1].row_stop)
        return ghost_start, ghost_stop


type OverlapPolicy[BufferedChunkT: BufferedChunk] = _OverlapPolicy[BufferedChunkT]


@dataclass(frozen=True)
class _CountSummary:
    rows: int


@dataclass(frozen=True)
class _RangeSummary:
    rows: int
    validation: RangeValidation
    # Empty or invalid input has no range endpoints.
    values: RangeValues | None


@dataclass(frozen=True)
class _PreparedInput:
    buffer: ChunkStore
    exchange: SparseAlltoall | None
    sources: tuple[int, ...]
    host_values: tuple[RangeValues | None, ...]


class OverlapInput[BufferedChunkT: BufferedChunk]:
    """Yield local input with the ghost rows required from other ranks."""

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        ch_in: Channel[TableChunk],
        policy: OverlapPolicy[BufferedChunkT],
        *,
        duplicated: bool,
        collective_id: int,
    ) -> None:
        self.context = context
        self.comm = comm
        self.ch_in = ch_in
        self.policy = policy
        self.multirank = comm.nranks != 1 and not duplicated
        self.collective_id = collective_id
        self._prepared: _PreparedInput | None = None

    async def __aenter__(self) -> OverlapInput[BufferedChunkT]:
        """Buffer and exchange input when it spans multiple ranks."""
        if self.multirank:
            self._prepared = await self._prepare()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Release messages still owned by this input."""
        prepared, self._prepared = self._prepared, None
        if prepared is not None:
            prepared.buffer.clear()

    def __aiter__(self) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
        """Yield prepared chunks in rolling evaluation order."""
        return self._chunks()

    async def _raw_owned_chunks(
        self,
    ) -> AsyncIterator[tuple[TableChunk, RangeValues | None]]:
        """Yield locally owned input chunks."""
        if not self.multirank:
            while (msg := await self.ch_in.recv(self.context)) is not None:
                yield TableChunk.from_message(msg, br=self.context.br()), None
        else:
            assert self._prepared is not None
            for position, msg in enumerate(self._prepared.buffer):
                yield (
                    TableChunk.from_message(msg, br=self.context.br()),
                    self._prepared.host_values[position]
                    if self._prepared.host_values
                    else None,
                )

    async def _raw_ghost_chunks(
        self, sources: Iterable[int]
    ) -> AsyncIterator[tuple[TableChunk, None]]:
        """Yield exchanged chunks from each source in rank order."""
        prepared = self._prepared
        for source in sources:
            assert prepared is not None
            assert prepared.exchange is not None
            for packed in prepared.exchange.extract(source):
                yield TableChunk.from_packed_data(packed, self.context.br()), None

    async def _chunks(self) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
        """Prepare owned and ghost chunks in logical row order."""
        if self.multirank and self._prepared is None:
            raise RuntimeError(
                "Multi-rank overlap input must be used as an async context manager"
            )
        row_offset = 0
        sources = self._prepared.sources if self._prepared else ()
        split = bisect_left(sources, self.comm.rank)
        inputs = (
            (False, self._raw_ghost_chunks(sources[:split])),
            (True, self._raw_owned_chunks()),
            (False, self._raw_ghost_chunks(sources[split:])),
        )
        for owned, values in inputs:
            async for raw_chunk, host_values in values:
                chunk = await self.policy.prepare_chunk(
                    self.context,
                    raw_chunk,
                    row_offset=row_offset,
                    host_values=host_values,
                )
                self.policy.observe(chunk)
                row_offset = chunk.row_stop
                yield StagedChunk(chunk, owned=owned)

    async def _prepare(self) -> _PreparedInput:
        buffer = ChunkStore(self.context)
        policy = self.policy
        try:
            if isinstance(policy, RangeOverlapPolicy):
                summary, host_values = await self._buffer_range_input(buffer, policy)
                gathered = await self._gather_ranges(summary)
                self._validate_ranges(policy, gathered)
                sends, sources, destinations = self._range_plan(summary, gathered)
                row_offset = sum(item.rows for item in gathered[: self.comm.rank])
                return await self._exchange(
                    buffer,
                    policy,
                    sends,
                    sources,
                    destinations,
                    row_offset,
                    host_values=host_values,
                    summaries=gathered,
                )

            if not isinstance(policy, RowCountOverlapPolicy):
                _unsupported_policy(policy)
            count_summary = await self._buffer_count_input(buffer)
            count_summaries = await self._gather_counts(count_summary)
            starts = tuple(
                accumulate((item.rows for item in count_summaries), initial=0)
            )
            sends, sources, destinations = self._count_plan(
                policy, tuple(pairwise(starts))
            )
            return await self._exchange(
                buffer,
                policy,
                sends,
                sources,
                destinations,
                starts[self.comm.rank],
            )
        except BaseException:
            buffer.clear()
            raise

    async def _buffer_count_input(self, buffer: ChunkStore) -> _CountSummary:
        rows = 0
        while (msg := await self.ch_in.recv(self.context)) is not None:
            chunk = TableChunk.from_message(msg, br=self.context.br())
            rows += chunk.shape[0]
            buffer.insert(Message(0, chunk))
        return _CountSummary(rows)

    async def _buffer_range_input(
        self,
        buffer: ChunkStore,
        policy: RangeOverlapPolicy,
    ) -> tuple[_RangeSummary, tuple[RangeValues | None, ...]]:
        row_offset = 0
        validation = RangeValidation.VALID
        summary_values = None
        host_values: list[RangeValues | None] = []
        while (msg := await self.ch_in.recv(self.context)) is not None:
            table_chunk = TableChunk.from_message(msg, br=self.context.br())
            nrows, _ = table_chunk.shape
            chunk = await policy.prepare_chunk(
                self.context,
                table_chunk,
                row_offset=row_offset,
                validate_index=False,
            )
            policy.observe(chunk)
            table_chunk = chunk.chunk
            values = chunk.overlap.values
            if nrows:
                status = policy.index_validation(
                    chunk.overlap.index_column, table_chunk.stream
                )
                if status is RangeValidation.VALID:
                    assert values is not None
                    if (
                        summary_values is not None
                        and values.first < summary_values.last
                    ):
                        status = RangeValidation.UNSORTED
                if validation is RangeValidation.VALID:
                    validation = status
                if status is RangeValidation.VALID:
                    assert values is not None
                    summary_values = RangeValues(
                        summary_values.first if summary_values else values.first,
                        values.last,
                        summary_values.lower if summary_values else values.lower,
                        values.upper,
                    )
            host_values.append(values)
            buffer.insert(Message(0, table_chunk))
            row_offset += nrows
        return _RangeSummary(row_offset, validation, summary_values), tuple(host_values)

    async def _allgather_ints(
        self, payload: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        fmt = f"<{len(payload)}q"
        gather = AllGather(self.context, self.comm, self.collective_id)
        try:
            gather.insert(
                0,
                PackedData.from_host_bytes(
                    struct.pack(fmt, *payload), self.context.br()
                ),
            )
        finally:
            gather.insert_finished()
        return tuple(
            struct.unpack(fmt, item.to_host_bytes())
            for item in await gather.extract_all(self.context, ordered=True)
        )

    async def _gather_counts(self, summary: _CountSummary) -> tuple[_CountSummary, ...]:
        return tuple(
            _CountSummary(item[0])
            for item in await self._allgather_ints((summary.rows,))
        )

    async def _gather_ranges(self, summary: _RangeSummary) -> tuple[_RangeSummary, ...]:
        values = summary.values
        validation = summary.validation
        if values is not None and any(
            not -(2**63) <= value < 2**63
            for value in (values.first, values.last, values.lower, values.upper)
        ):
            validation = RangeValidation.OUT_OF_BOUNDS
        packed_values = (
            (values.first, values.last, values.lower, values.upper)
            if values is not None and validation is RangeValidation.VALID
            else (0, 0, 0, 0)
        )
        return tuple(
            _RangeSummary(
                item[0],
                RangeValidation(item[1]),
                RangeValues(*item[2:])
                if item[0] and item[1] == RangeValidation.VALID
                else None,
            )
            for item in await self._allgather_ints(
                (summary.rows, int(validation), *packed_values)
            )
        )

    @staticmethod
    def _validate_ranges(
        policy: RangeOverlapPolicy,
        summaries: Sequence[_RangeSummary],
    ) -> None:
        validations = {item.validation for item in summaries}
        if RangeValidation.OUT_OF_BOUNDS in validations:
            raise NotImplementedError(
                "Multi-rank rolling does not support index values outside the "
                "signed 64-bit range"
            )
        if RangeValidation.NULL in validations:
            policy.raise_validation(RangeValidation.NULL)
        previous = None
        for item in summaries:
            if item.validation is RangeValidation.UNSORTED:
                break
            if not item.rows:
                continue
            values = item.values
            if (
                values is None
                or values.first > values.last
                or (previous is not None and values.first < previous)
            ):
                break
            previous = values.last
        else:
            return
        policy.raise_validation(RangeValidation.UNSORTED)

    @staticmethod
    def _intersection(left: RowRange, right: RowRange) -> RowRange | None:
        start, stop = max(left[0], right[0]), min(left[1], right[1])
        return (start, stop) if start < stop else None

    def _count_plan(
        self,
        policy: RowCountOverlapPolicy,
        spans: Sequence[RowRange],
    ) -> tuple[list[GhostSend], tuple[int, ...], tuple[int, ...]]:
        total = spans[-1][1] if spans else 0
        requests = tuple(
            tuple(
                span
                for span in (
                    (max(0, start - policy.preceding), start),
                    (stop, min(total, stop + policy.following)),
                )
                if start != stop and span[0] < span[1]
            )
            for start, stop in spans
        )
        rank = self.comm.rank
        sends: list[GhostSend] = []
        sources: list[int] = []
        destinations: set[int] = set()
        for other, span in enumerate(spans):
            if other == rank:
                continue
            if any(self._intersection(span, request) for request in requests[rank]):
                sources.append(other)
            for request in requests[other]:
                if (piece := self._intersection(spans[rank], request)) is not None:
                    sends.append((other, *piece))
                    destinations.add(other)
        return sends, tuple(sources), tuple(destinations)

    def _range_plan(
        self,
        local: _RangeSummary,
        summaries: Sequence[_RangeSummary],
    ) -> tuple[list[GhostSend], tuple[int, ...], tuple[int, ...]]:
        rank = self.comm.rank
        local_values = local.values
        sources = tuple(
            source
            for source, item in enumerate(summaries)
            if source != rank
            and local_values is not None
            and item.values is not None
            and item.values.last >= local_values.lower
            and item.values.first <= local_values.upper
        )
        destinations = tuple(
            destination
            for destination, item in enumerate(summaries)
            if destination != rank
            and local_values is not None
            and item.values is not None
            and local_values.last >= item.values.lower
            and local_values.first <= item.values.upper
        )
        return [], sources, destinations

    async def _exchange(
        self,
        buffer: ChunkStore,
        policy: RangeOverlapPolicy | RowCountOverlapPolicy,
        sends: Sequence[GhostSend],
        sources: Sequence[int],
        destinations: Sequence[int],
        row_offset: int,
        *,
        host_values: tuple[RangeValues | None, ...] = (),
        summaries: Sequence[_RangeSummary] = (),
    ) -> _PreparedInput:
        sources = tuple(sorted(sources))
        destinations = tuple(sorted(set(destinations)))
        if not (sources or destinations):
            return _PreparedInput(buffer, None, sources, host_values)
        exchange = SparseAlltoall(
            self.context,
            self.comm,
            self.collective_id,
            srcs=sources,
            dsts=destinations,
        )
        source, output = buffer, ChunkStore(self.context)
        start = 0
        try:
            for position, msg in enumerate(source):
                chunk = TableChunk.from_message(msg, br=self.context.br())
                size = chunk.shape[0]
                pieces: list[tuple[int, int, int]] = []
                range_values = None
                if isinstance(policy, RangeOverlapPolicy) and size:
                    range_values = host_values[position]
                    assert range_values is not None
                    range_requests = tuple(
                        (destination, destination_values)
                        for destination in destinations
                        if (destination_values := summaries[destination].values)
                        is not None
                        and range_values.last >= destination_values.lower
                        and range_values.first <= destination_values.upper
                    )
                else:
                    span = (start + row_offset, start + row_offset + size)
                    pieces = [
                        (destination, piece[0] - span[0], piece[1] - span[0])
                        for destination, request_start, request_stop in sends
                        if (
                            piece := self._intersection(
                                span, (request_start, request_stop)
                            )
                        )
                        is not None
                    ]
                    range_requests = ()
                if not range_requests and not pieces:
                    output.insert(Message(0, chunk))
                    start += size
                    continue
                available, extra = await make_table_chunks_available_or_wait(
                    self.context,
                    chunk,
                    reserve_extra=0,
                    net_memory_delta=0,
                )
                with opaque_memory_usage(extra):
                    if (
                        isinstance(policy, RangeOverlapPolicy)
                        and range_values is not None
                    ):
                        pieces = await self._range_pieces(
                            policy,
                            available,
                            size,
                            range_values,
                            range_requests,
                        )
                    for destination, begin, end in pieces:
                        await self._send(exchange, destination, available, begin, end)
                    output.insert(Message(0, available))
                start += size
        except BaseException:
            output.clear()
            raise
        finally:
            try:
                source.clear()
            except BaseException:
                output.clear()
                raise
            finally:
                try:
                    await exchange.insert_finished(self.context)
                except BaseException:
                    output.clear()
                    raise
        return _PreparedInput(output, exchange, sources, host_values)

    async def _range_pieces(
        self,
        policy: RangeOverlapPolicy,
        chunk: TableChunk,
        size: int,
        local: RangeValues,
        requests: Sequence[RangeRequest],
    ) -> list[tuple[int, int, int]]:
        pieces = []
        index = chunk.table_view().columns()[policy.index]
        for destination, values in requests:
            lower, upper = values.lower, values.upper
            begin = (
                0
                if lower < local.first or (policy.start_closed and lower == local.first)
                else await policy.insertion_point(
                    self.context,
                    index,
                    lower,
                    policy.find_start,
                    chunk.stream,
                )
            )
            end = (
                size
                if upper > local.last or (policy.end_closed and upper == local.last)
                else await policy.insertion_point(
                    self.context,
                    index,
                    upper,
                    policy.find_end,
                    chunk.stream,
                )
            )
            if begin < end:
                pieces.append((destination, begin, end))
        return pieces

    async def _send(
        self,
        exchange: SparseAlltoall,
        destination: int,
        chunk: TableChunk,
        start: int,
        stop: int,
    ) -> None:
        table = (
            plc.Table(chunk.table_view().columns())
            if start == 0 and stop == chunk.shape[0]
            else plc.copying.slice(
                chunk.table_view(), [start, stop], stream=chunk.stream
            )[0]
        )
        piece = TableChunk.from_pylibcudf_table(
            table,
            chunk.stream,
            exclusive_view=False,
            br=self.context.br(),
        )
        piece, extra = await make_table_chunks_available_or_wait(
            self.context,
            piece,
            reserve_extra=piece.into_packed_data_cost(),
            net_memory_delta=0,
        )
        exchange.insert(destination, piece.into_packed_data(extra))
