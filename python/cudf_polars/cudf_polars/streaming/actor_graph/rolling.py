# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import struct
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.ir import IR, Rolling
from cudf_polars.dsl.utils.windows import duration_to_scalar
from cudf_polars.streaming.actor_graph.collectives.overlap import (
    BufferedInput,
    GhostSend,
    RowExchange,
)
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    evaluate_chunk,
    gather_in_task_group,
    maybe_remap_partitioning,
    names_to_indices,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.rolling import FixedSizeRolling
from cudf_polars.utils.cuda_stream import join_cuda_streams, stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer


RowRange = tuple[int, int]
RangeValues = tuple[Any, Any, Any, Any]


@dataclass
class BufferedChunk:
    """An input chunk with its global row span."""

    sequence_number: int
    chunk: TableChunk
    row_start: int
    num_rows: int

    @property
    def row_stop(self) -> int:
        """Global row offset immediately after this chunk."""
        return self.row_start + self.num_rows


@dataclass
class RangeOverlap:
    """Range-window metadata for one buffered input chunk."""

    index_column: plc.Column
    lower_bound_column: plc.Column
    upper_bound_column: plc.Column
    first: Any | None
    last: Any | None
    lower_bound: Any | None
    upper_bound: Any | None


@dataclass
class RangeBufferedChunk(BufferedChunk):
    """A buffered chunk with range-window overlap metadata."""

    overlap: RangeOverlap


@dataclass
class StagedChunk[BufferedChunkT: BufferedChunk]:
    """A prepared chunk and its output sequence number when locally owned."""

    chunk: BufferedChunkT
    output_sequence_number: int | None = None


@dataclass(frozen=True)
class RangeChunkBounds:
    """Host-side bounds for one rank-local input chunk."""

    row_start: int
    row_stop: int
    first: Any
    last: Any


@dataclass(frozen=True)
class RangeInputSummary:
    """Compact rank-local state used to plan range overlap."""

    local_rows: int
    lower_value: int | None
    upper_value: int | None
    first_value: int | None
    last_value: int | None
    chunk_bounds: tuple[RangeChunkBounds, ...]

    @property
    def has_rows(self) -> bool:
        """Return whether this rank has any rows."""
        return bool(self.chunk_bounds)


@dataclass(frozen=True)
class RangeRequests:
    """Range-overlap requests gathered from every rank."""

    row_counts: tuple[int, ...]
    lower_values: tuple[Any | None, ...]
    upper_values: tuple[Any | None, ...]
    first_values: tuple[Any | None, ...]
    last_values: tuple[Any | None, ...]


class RollingInput[BufferedChunkT: BufferedChunk]:
    """Direct local input or buffered multi-rank input with ghost rows."""

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        ch_in: Channel[TableChunk],
        policy: _RollingPolicy[BufferedChunkT],
        metadata: ChannelMetadata,
        *,
        collective_id: int,
    ) -> None:
        self.context = context
        self.comm = comm
        self.ch_in = ch_in
        self.policy = policy
        self.collective_id = collective_id
        self.multirank = comm.nranks != 1 and not metadata.duplicated
        self._buffer: BufferedInput | None = None
        self._exchange: RowExchange | None = None
        self._replay_channel = context.create_channel() if self.multirank else None
        self._staged: dict[int, int | None] = {}
        self._range_values: dict[int, RangeValues] = {}

    async def __aenter__(self) -> RollingInput[BufferedChunkT]:
        """Prepare ghost exchange when input spans multiple ranks."""
        try:
            if self.multirank:
                await self._prepare_multirank()
        except BaseException:
            if self._buffer is not None:
                self._buffer.clear()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Release any messages still owned by this input."""
        if exc_info[0] is not None and self._replay_channel is not None:
            await self._replay_channel.shutdown(self.context)
        if self._buffer is not None:
            self._buffer.clear()

    def producer_tasks(self) -> tuple[Any, ...]:
        """Return producer coroutines required by this input."""
        return (self._replay(),) if self.multirank else ()

    def __aiter__(self) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
        """Yield prepared chunks in rolling evaluation order."""
        return self._chunks()

    async def _chunks(self) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
        channel = self._replay_channel if self.multirank else self.ch_in
        assert channel is not None
        row_offset = 0
        while (msg := await channel.recv(self.context)) is not None:
            output_sequence_number = (
                self._staged.pop(msg.sequence_number)
                if self.multirank
                else msg.sequence_number
            )
            if (
                self.multirank
                and isinstance(self.policy, RangeOverlapPolicy)
                and output_sequence_number in self._range_values
            ):
                sequence_number = msg.sequence_number
                table_chunk = TableChunk.from_message(msg, br=self.context.br())
                chunk = cast(
                    "BufferedChunkT",
                    await self.policy.prepare_table_chunk(
                        self.context,
                        sequence_number,
                        table_chunk,
                        row_offset=row_offset,
                        host_values=self._range_values[output_sequence_number],
                    ),
                )
            else:
                chunk = await self.policy.prepare_chunk(
                    self.context, msg, row_offset=row_offset
                )
            self.policy.observe(chunk)
            row_offset = chunk.row_stop
            yield StagedChunk(chunk, output_sequence_number)

    async def _prepare_multirank(self) -> None:
        self._buffer = BufferedInput(self.context)
        if isinstance(self.policy, RangeOverlapPolicy):
            summary = await self._buffer_range_input()
            requests = await _gather_range_requests(
                self.context,
                self.comm,
                summary,
                collective_id=self.collective_id,
            )
            _validate_global_range_order(requests, self.policy)
            starts = _row_starts(requests.row_counts)
            owned = (starts[self.comm.rank], starts[self.comm.rank + 1])
            sources = _range_sources(self.comm.rank, requests)
            destinations = _range_destinations(self.comm.rank, summary, requests)
            sends = await _range_sends(
                self.context,
                self._buffer,
                self.policy,
                summary,
                requests,
                destinations,
                owned[0],
            )
        else:
            assert isinstance(self.policy, RowCountOverlapPolicy)
            local_rows = await self._buffer_count_input()
            counts = tuple(
                values[0]
                for values in await _allgather_int64s(
                    self.context,
                    self.comm,
                    local_rows,
                    collective_id=self.collective_id,
                )
            )
            starts = _row_starts(counts)
            spans = tuple((starts[i], starts[i + 1]) for i in range(len(counts)))
            owned = spans[self.comm.rank]
            count_requests = _count_requests(spans, self.policy)
            sends, sources, destinations = _count_routing(
                self.comm.rank, spans, count_requests
            )
        self._buffer.set_row_offset(owned[0])
        self._exchange = RowExchange(
            self.context,
            self.comm,
            self._buffer,
            sends,
            sources,
            destinations,
            collective_id=self.collective_id,
        )
        await self._exchange.prepare()

    async def _buffer_count_input(self) -> int:
        assert self._buffer is not None
        row_offset = 0
        while (msg := await self.ch_in.recv(self.context)) is not None:
            sequence_number = msg.sequence_number
            chunk = TableChunk.from_message(msg, br=self.context.br())
            nrows, _ = chunk.shape
            self._buffer.insert(
                Message(sequence_number, chunk),
                row_start=row_offset,
                num_rows=nrows,
            )
            row_offset += nrows
        return row_offset

    async def _buffer_range_input(self) -> RangeInputSummary:
        assert self._buffer is not None
        assert isinstance(self.policy, RangeOverlapPolicy)
        row_offset = 0
        bounds: list[RangeChunkBounds] = []
        lower = upper = first = last = None
        while (msg := await self.ch_in.recv(self.context)) is not None:
            sequence_number = msg.sequence_number
            chunk = await self.policy.prepare_chunk(
                self.context, msg, row_offset=row_offset
            )
            self.policy.observe(chunk)
            row_offset = chunk.row_stop
            if chunk.num_rows != 0:
                overlap = chunk.overlap
                assert overlap.first is not None
                assert overlap.last is not None
                bounds.append(
                    RangeChunkBounds(
                        chunk.row_start, chunk.row_stop, overlap.first, overlap.last
                    )
                )
                if lower is None:
                    lower = overlap.lower_bound
                    first = overlap.first
                upper = overlap.upper_bound
                last = overlap.last
                assert overlap.lower_bound is not None
                assert overlap.upper_bound is not None
                self._range_values[sequence_number] = (
                    overlap.first,
                    overlap.last,
                    overlap.lower_bound,
                    overlap.upper_bound,
                )
            self._buffer.insert(
                Message(sequence_number, chunk.chunk),
                row_start=chunk.row_start,
                num_rows=chunk.num_rows,
            )
        return RangeInputSummary(row_offset, lower, upper, first, last, tuple(bounds))

    async def _replay(self) -> None:
        assert self._buffer is not None
        assert self._exchange is not None
        assert self._replay_channel is not None
        replay_channel = self._replay_channel
        sequence_number = 0

        async def send(chunk: TableChunk, output: int | None = None) -> None:
            nonlocal sequence_number
            self._staged[sequence_number] = output
            await replay_channel.send(self.context, Message(sequence_number, chunk))
            sequence_number += 1

        for source in self._exchange.sources:
            if source < self.comm.rank:
                for chunk in self._exchange.receive(source):
                    await send(chunk)
        for msg in self._buffer.replay():
            output_sequence_number = msg.sequence_number
            await send(
                TableChunk.from_message(msg, br=self.context.br()),
                output_sequence_number,
            )
        for source in self._exchange.sources:
            if source > self.comm.rank:
                for chunk in self._exchange.receive(source):
                    await send(chunk)
        await replay_channel.drain(self.context)


@dataclass
class RollingManager:
    """Rolling policy and rank-agnostic local executor."""

    context: Context
    ir: Rolling | FixedSizeRolling
    ir_context: IRExecutionContext
    policy: _RollingPolicy[Any] = field(init=False)

    def __post_init__(self) -> None:
        """Choose the policy for this rolling node."""
        if isinstance(self.ir, Rolling):
            self.policy = RangeOverlapPolicy.from_ir(
                self.ir, self.ir_context.get_cuda_stream()
            )
        else:
            self.policy = RowCountOverlapPolicy(
                self.ir.preceding_overlap, self.ir.following_overlap
            )

    def __enter__(self) -> RollingManager:
        """Enter the rolling manager context."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release policy-owned resources."""
        self.policy.close()

    @staticmethod
    def _latest_nonempty_chunk(current: Any, future: list[StagedChunk[Any]]) -> Any:
        for staged in reversed(future):
            if staged.chunk.num_rows != 0:
                return staged.chunk
        return current

    async def _fill_future(
        self,
        chunks: AsyncIterator[StagedChunk[Any]],
        current: Any,
        future: list[StagedChunk[Any]],
    ) -> bool:
        """Read chunks until current has enough leading context."""
        while not self.policy.has_complete_future(
            self._latest_nonempty_chunk(current, future),
            current,
            self.context,
        ):
            chunk = await anext(chunks, None)
            if chunk is None:
                return True
            future.append(chunk)
        return False

    async def execute(
        self,
        ch_out: Channel[TableChunk],
        tracer: ActorTracer | None,
        rolling_input: RollingInput[Any],
    ) -> None:
        """Evaluate this rolling node over its prepared input stream."""
        history: list[Any] = []
        future: list[StagedChunk[Any]] = []
        input_exhausted = False
        chunks = aiter(rolling_input)
        staged = await anext(chunks, None)
        while staged is not None:
            cursor = staged.chunk
            self.policy.validate_cursor(self.context, cursor)
            if staged.output_sequence_number is None:
                history.append(cursor)
            elif cursor.num_rows == 0:
                result = await evaluate_chunk(
                    self.context,
                    cursor.chunk,
                    self.ir,
                    ir_context=self.ir_context,
                    already_available=True,
                )
                await send_chunk(
                    self.context,
                    ch_out,
                    result,
                    staged.output_sequence_number,
                    tracer=tracer,
                )
            else:
                history = self.policy.evict_history(history, cursor, self.context)
                if not input_exhausted:
                    input_exhausted = await self._fill_future(
                        chunks,
                        cursor,
                        future,
                    )
                result = await self.policy.evaluate_cursor(
                    self.context,
                    self.ir,
                    self.ir_context,
                    cursor,
                    history=history,
                    future=[item.chunk for item in future],
                )
                retained = await self.policy.history_chunk(
                    self.context,
                    cursor,
                    ir_context=self.ir_context,
                )
                if retained is not None:
                    history.append(retained)
                await send_chunk(
                    self.context,
                    ch_out,
                    result,
                    staged.output_sequence_number,
                    tracer=tracer,
                )
            if future:
                staged = future.pop(0)
            else:
                staged = await anext(chunks, None)
        await ch_out.drain(self.context)


class _RollingPolicy[BufferedChunkT: BufferedChunk]:
    """Base policy for evaluating ghost-expanded cursor chunks."""

    def observe(self, chunk: BufferedChunkT) -> None:
        del chunk

    def close(self) -> None:
        pass

    def validate_cursor(self, context: Context, cursor: BufferedChunkT) -> None:
        del context, cursor

    def evict_history(
        self,
        history: list[BufferedChunkT],
        cursor: BufferedChunkT,
        context: Context,
    ) -> list[BufferedChunkT]:
        raise NotImplementedError

    def has_complete_future(
        self,
        latest: BufferedChunkT,
        current: BufferedChunkT,
        context: Context,
    ) -> bool:
        raise NotImplementedError

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunkT,
        *,
        ir_context: IRExecutionContext,
    ) -> BufferedChunkT | None:
        """Return the cursor rows that future cursors may need."""
        del context, ir_context
        return cursor

    async def evaluate_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: BufferedChunkT,
        *,
        history: list[BufferedChunkT],
        future: list[BufferedChunkT],
    ) -> TableChunk:
        raise NotImplementedError

    async def prepare_chunk(
        self,
        context: Context,
        msg: Message,
        *,
        row_offset: int,
    ) -> BufferedChunkT:
        raise NotImplementedError

    async def _extract_region(
        self,
        context: Context,
        input_chunks: Sequence[BufferedChunk],
        row_start: int,
        row_stop: int,
        *,
        ir_context: IRExecutionContext,
    ) -> TableChunk:
        """Slice all buffered chunks intersecting a global row range."""
        chunks: list[TableChunk] = []
        for buf in input_chunks:
            if buf.row_start >= row_stop:
                break
            start = max(row_start, buf.row_start)
            stop = min(row_stop, buf.row_stop)
            if start < stop:
                if start == buf.row_start and stop == buf.row_stop:
                    chunks.append(buf.chunk)
                else:
                    chunks.append(
                        TableChunk.from_pylibcudf_table(
                            plc.copying.slice(
                                buf.chunk.table_view(),
                                # Translate to chunk-local row coordinates
                                [
                                    start - buf.row_start,
                                    stop - buf.row_start,
                                ],
                                stream=buf.chunk.stream,
                            )[0],
                            buf.chunk.stream,
                            exclusive_view=False,
                            br=context.br(),
                        )
                    )
        assert chunks, "Should have found at least one chunk to extract from"
        if len(chunks) == 1:
            return chunks[0]
        reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
            sum(chunk.data_alloc_size() for chunk in chunks), net_memory_delta=0
        )
        chunk_streams = [chunk.stream for chunk in chunks]
        with (
            opaque_memory_usage(reservation),
            stream_ordered_after(ir_context.get_cuda_stream, chunk_streams) as stream,
        ):
            table = plc.concatenate.concatenate(
                [chunk.table_view() for chunk in chunks],
                stream=stream,
                mr=context.br().device_mr,
            )
            return TableChunk.from_pylibcudf_table(
                table, stream=stream, exclusive_view=True, br=context.br()
            )

    async def _evaluate_ghosted_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: BufferedChunk,
        *,
        chunks: Sequence[BufferedChunk],
        ghost_start: int,
        ghost_stop: int,
    ) -> TableChunk:
        """Evaluate a cursor chunk with surrounding ghost rows."""
        if ghost_start == cursor.row_start and ghost_stop == cursor.row_stop:
            return await evaluate_chunk(
                context,
                cursor.chunk,
                ir,
                ir_context=ir_context,
                already_available=True,
            )
        ghosted_chunk = await self._extract_region(
            context, chunks, ghost_start, ghost_stop, ir_context=ir_context
        )
        result = await evaluate_chunk(
            context,
            ghosted_chunk,
            ir,
            ir_context=ir_context,
            already_available=True,
        )
        (table,) = plc.copying.slice(
            result.table_view(),
            [cursor.row_start - ghost_start, cursor.row_stop - ghost_start],
            stream=result.stream,
        )
        reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
            result.data_alloc_size(), net_memory_delta=0
        )
        with opaque_memory_usage(reservation):
            table = table.copy(result.stream, context.br().device_mr)
        return TableChunk.from_pylibcudf_table(
            table,
            result.stream,
            exclusive_view=True,
            br=context.br(),
        )


@dataclass
class RangeOverlapPolicy(_RollingPolicy[RangeBufferedChunk]):
    """Overlap policy for range-based rolling windows."""

    lower: plc.Scalar
    upper: plc.Scalar
    index: int
    index_dtype: plc.DataType
    find_start: Callable[..., plc.Column]
    find_end: Callable[..., plc.Column]
    start_closed: bool
    end_closed: bool
    stream: Stream
    index_name: str
    observed_streams: set[Stream] = field(default_factory=set)
    previous_index_value: Any | None = None

    @classmethod
    def from_ir(cls, ir: Rolling, stream: Stream) -> RangeOverlapPolicy:
        """Create reusable range-bound state for the rolling actor."""
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
    ) -> Any:
        """Copy a single-row ordering value to host."""
        if plc.traits.is_chrono(value.type()):
            value = plc.unary.bit_cast(
                value,
                cls._chrono_storage_dtype(value.type()),
                stream=stream,
                mr=br.device_mr,
            )
        return value.to_scalar(stream=stream).to_py(stream=stream)

    @classmethod
    def _host_ordering_value_at(
        cls,
        column: plc.Column,
        row: int,
        *,
        stream: Stream,
        br: BufferResource,
    ) -> Any:
        """Copy one ordering value from a device column to host."""
        (value,) = plc.copying.slice(column, [row, row + 1], stream=stream)
        return cls._host_ordering_value(value, stream=stream, br=br)

    @staticmethod
    async def _insertion_point(
        context: Context,
        index_column: plc.Column,
        needle: plc.Column,
        find: Callable[..., plc.Column],
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
                    stream=stream,
                    mr=context.br().device_mr,
                )
                .to_scalar(stream=stream)
                .to_py(stream=stream)
            )
        assert isinstance(value, int)
        return value

    @staticmethod
    async def _global_insertion_row(
        chunks: Sequence[RangeBufferedChunk],
        needle: plc.Column,
        needle_value: Any,
        find: Callable[..., plc.Column],
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
            assert chunk.overlap.last is not None
            if (
                chunk.overlap.last <= needle_value
                if upper_bound
                else chunk.overlap.last < needle_value
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

    def validate_index_column(self, index_column: plc.Column, stream: Stream) -> None:
        """Validate chunk-local ordering before using search primitives."""
        if index_column.null_count() != 0:
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling may not contain nulls"
            )
        if index_column.size() > 1 and not plc.sorting.is_sorted(
            plc.Table([index_column]),
            [plc.types.Order.ASCENDING],
            [plc.types.NullOrder.BEFORE],
            stream=stream,
        ):
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling is not sorted, "
                "please sort first"
            )

    def validate_cursor(self, context: Context, cursor: RangeBufferedChunk) -> None:
        """Check that the index remains sorted across chunk boundaries."""
        del context
        if cursor.num_rows == 0:
            return
        assert cursor.overlap.first is not None
        assert cursor.overlap.last is not None
        if (
            self.previous_index_value is not None
            and cursor.overlap.first < self.previous_index_value
        ):
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling is not sorted, "
                "please sort first"
            )
        self.previous_index_value = cursor.overlap.last

    async def prepare_chunk(
        self,
        context: Context,
        msg: Message,
        *,
        row_offset: int,
    ) -> RangeBufferedChunk:
        """Convert a message to a staged chunk and extract range metadata."""
        sequence_number = msg.sequence_number
        chunk = TableChunk.from_message(msg, br=context.br())
        return await self.prepare_table_chunk(
            context,
            sequence_number,
            chunk,
            row_offset=row_offset,
        )

    async def prepare_table_chunk(
        self,
        context: Context,
        sequence_number: int,
        chunk: TableChunk,
        *,
        row_offset: int,
        already_available: bool = False,
        host_values: RangeValues | None = None,
    ) -> RangeBufferedChunk:
        """Extract range metadata from a table chunk."""
        nrows, _ = chunk.shape
        if already_available:
            extra = None
        else:
            chunk, extra = await make_table_chunks_available_or_wait(
                context,
                chunk,
                reserve_extra=32,
                net_memory_delta=0,
            )
        memory_scope = nullcontext() if extra is None else opaque_memory_usage(extra)
        with memory_scope:
            index_column = chunk.table_view().columns()[self.index]
            if host_values is None:
                self.validate_index_column(index_column, chunk.stream)
            if nrows == 0:
                overlap = RangeOverlap(
                    index_column, index_column, index_column, None, None, None, None
                )
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
                if host_values is None:
                    host_values = (
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
                    *host_values,
                )
        return RangeBufferedChunk(sequence_number, chunk, row_offset, nrows, overlap)

    def evict_history(
        self,
        history: list[RangeBufferedChunk],
        cursor: RangeBufferedChunk,
        context: Context,
    ) -> list[RangeBufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        del context
        if not history:
            return []
        assert cursor.overlap.lower_bound is not None
        if self.start_closed:
            return [
                chunk
                for chunk in history
                if chunk.overlap.last is not None
                and chunk.overlap.last >= cursor.overlap.lower_bound
            ]
        return [
            chunk
            for chunk in history
            if chunk.overlap.last is not None
            and chunk.overlap.last > cursor.overlap.lower_bound
        ]

    def has_complete_future(
        self,
        latest: RangeBufferedChunk,
        current: RangeBufferedChunk,
        context: Context,
    ) -> bool:
        """Return whether latest contains current's upper insertion point."""
        del context
        if latest.overlap.last is None:
            return False
        assert current.overlap.upper_bound is not None
        if self.end_closed:
            return latest.overlap.last > current.overlap.upper_bound
        return latest.overlap.last >= current.overlap.upper_bound

    async def evaluate_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: RangeBufferedChunk,
        *,
        history: list[RangeBufferedChunk],
        future: list[RangeBufferedChunk],
    ) -> TableChunk:
        """Evaluate the rolling aggregation for the cursor chunk."""
        chunks = [*history, cursor, *future]
        assert cursor.overlap.lower_bound is not None
        assert cursor.overlap.upper_bound is not None
        ghost_start = await self._global_insertion_row(
            chunks,
            cursor.overlap.lower_bound_column,
            cursor.overlap.lower_bound,
            self.find_start,
            upper_bound=not self.start_closed,
            dtype=self.index_dtype,
            context=context,
            needle_stream=cursor.chunk.stream,
        )
        ghost_stop = await self._global_insertion_row(
            chunks,
            cursor.overlap.upper_bound_column,
            cursor.overlap.upper_bound,
            self.find_end,
            upper_bound=self.end_closed,
            dtype=self.index_dtype,
            context=context,
            needle_stream=cursor.chunk.stream,
        )
        # We must extract at least the whole of the current cursor chunk.
        ghost_start = min(cursor.row_start, ghost_start)
        ghost_stop = max(cursor.row_stop, ghost_stop)
        return await self._evaluate_ghosted_cursor(
            context,
            ir,
            ir_context,
            cursor,
            chunks=chunks,
            ghost_start=ghost_start,
            ghost_stop=ghost_stop,
        )


@dataclass
class RowCountOverlapPolicy(_RollingPolicy[BufferedChunk]):
    """Overlap policy for fixed-size rolling expressions."""

    preceding: int
    following: int

    async def prepare_chunk(
        self,
        context: Context,
        msg: Message,
        *,
        row_offset: int,
    ) -> BufferedChunk:
        """Convert a message to an available chunk with row-span metadata."""
        chunk = TableChunk.from_message(msg, br=context.br())
        nrows, _ = chunk.shape
        chunk, _ = await make_table_chunks_available_or_wait(
            context, chunk, reserve_extra=0, net_memory_delta=0
        )
        return BufferedChunk(msg.sequence_number, chunk, row_offset, nrows)

    def evict_history(
        self,
        history: list[BufferedChunk],
        cursor: BufferedChunk,
        context: Context,
    ) -> list[BufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        del context
        ghost_start = max(0, cursor.row_start - self.preceding)
        return [chunk for chunk in history if chunk.row_stop > ghost_start]

    def has_complete_future(
        self,
        latest: BufferedChunk,
        current: BufferedChunk,
        context: Context,
    ) -> bool:
        """Return whether latest contains enough leading rows."""
        del context
        return latest.row_stop >= current.row_stop + self.following

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunk,
        *,
        ir_context: IRExecutionContext,
    ) -> BufferedChunk | None:
        """Keep only emitted rows that a future fixed-size window can use."""
        del ir_context
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
            cursor.sequence_number,
            TableChunk.from_pylibcudf_table(
                table,
                stream,
                exclusive_view=True,
                br=context.br(),
            ),
            start,
            cursor.row_stop - start,
        )

    async def evaluate_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: BufferedChunk,
        *,
        history: list[BufferedChunk],
        future: list[BufferedChunk],
    ) -> TableChunk:
        """Evaluate fixed-size rolling expressions for the cursor chunk."""
        chunks = [*history, cursor, *future]
        ghost_start = max(0, cursor.row_start - self.preceding)
        ghost_stop = min(cursor.row_stop + self.following, chunks[-1].row_stop)
        return await self._evaluate_ghosted_cursor(
            context,
            ir,
            ir_context,
            cursor,
            chunks=chunks,
            ghost_start=ghost_start,
            ghost_stop=ghost_stop,
        )


def _row_starts(row_counts: Sequence[int]) -> list[int]:
    starts = [0]
    for count in row_counts:
        starts.append(starts[-1] + count)
    return starts


def _intersection(left: RowRange, right: RowRange) -> RowRange | None:
    start = max(left[0], right[0])
    stop = min(left[1], right[1])
    return (start, stop) if start < stop else None


def _count_requests(
    spans: Sequence[RowRange], policy: RowCountOverlapPolicy
) -> tuple[tuple[RowRange, ...], ...]:
    total_rows = spans[-1][1] if spans else 0
    requests: list[tuple[RowRange, ...]] = []
    for start, stop in spans:
        ghosts: list[RowRange] = []
        if start != stop and policy.preceding:
            ghosts.append((max(0, start - policy.preceding), start))
        if start != stop and policy.following:
            ghosts.append((stop, min(total_rows, stop + policy.following)))
        requests.append(tuple(span for span in ghosts if span[0] < span[1]))
    return tuple(requests)


def _count_routing(
    rank: int,
    spans: Sequence[RowRange],
    requests: Sequence[Sequence[RowRange]],
) -> tuple[list[GhostSend], tuple[int, ...], tuple[int, ...]]:
    sends: list[GhostSend] = []
    sources: list[int] = []
    destinations: list[int] = []
    for other, span in enumerate(spans):
        if other == rank:
            continue
        if any(_intersection(span, request) for request in requests[rank]):
            sources.append(other)
        for request in requests[other]:
            if (piece := _intersection(spans[rank], request)) is not None:
                sends.append(GhostSend(other, *piece))
                if other not in destinations:
                    destinations.append(other)
    return sends, tuple(sources), tuple(destinations)


def _single_ordering_value_column(
    value: Any,
    dtype: plc.DataType,
    stream: Stream,
    br: BufferResource,
) -> plc.Column:
    if plc.traits.is_chrono(dtype):
        storage = plc.Column.from_scalar(
            plc.Scalar.from_py(
                value, RangeOverlapPolicy._chrono_storage_dtype(dtype), stream=stream
            ),
            1,
            stream=stream,
        )
        return plc.unary.bit_cast(storage, dtype, stream=stream, mr=br.device_mr)
    return plc.Column.from_scalar(
        plc.Scalar.from_py(value, dtype, stream=stream), 1, stream=stream
    )


async def _allgather_int64s(
    context: Context,
    comm: Communicator,
    *values: int,
    collective_id: int,
) -> tuple[tuple[int, ...], ...]:
    """Gather a fixed number of signed integers from every rank."""
    fmt = f"<{len(values)}q"
    gather = AllGather(context, comm, collective_id)
    try:
        gather.insert(
            0, PackedData.from_host_bytes(struct.pack(fmt, *values), context.br())
        )
    finally:
        gather.insert_finished()
    return tuple(
        struct.unpack(fmt, packed.to_host_bytes())
        for packed in await gather.extract_all(context, ordered=True)
    )


async def _gather_range_requests(
    context: Context,
    comm: Communicator,
    summary: RangeInputSummary,
    *,
    collective_id: int,
) -> RangeRequests:
    values = await _allgather_int64s(
        context,
        comm,
        summary.local_rows,
        int(summary.has_rows),
        int(summary.lower_value or 0),
        int(summary.upper_value or 0),
        int(summary.first_value or 0),
        int(summary.last_value or 0),
        collective_id=collective_id,
    )
    row_counts = tuple(item[0] for item in values)
    raw_lower = tuple(item[2] if item[1] else None for item in values)
    raw_upper = tuple(item[3] if item[1] else None for item in values)
    first_values = tuple(item[4] if item[1] else None for item in values)
    last_values = tuple(item[5] if item[1] else None for item in values)
    return RangeRequests(
        row_counts,
        raw_lower,
        raw_upper,
        first_values,
        last_values,
    )


def _validate_global_range_order(
    requests: RangeRequests, policy: RangeOverlapPolicy
) -> None:
    previous_last = None
    for rank, row_count in enumerate(requests.row_counts):
        if not row_count:
            continue
        first = requests.first_values[rank]
        last = requests.last_values[rank]
        if (
            first is None
            or last is None
            or first > last
            or (previous_last is not None and first < previous_last)
        ):
            raise RuntimeError(
                f"Index column '{policy.index_name}' in rolling is not sorted, "
                "please sort first"
            )
        previous_last = last


def _range_sources(rank: int, requests: RangeRequests) -> tuple[int, ...]:
    lower = requests.lower_values[rank]
    upper = requests.upper_values[rank]
    if lower is None or upper is None:
        return ()
    return tuple(
        source
        for source in range(len(requests.row_counts))
        if source != rank
        and requests.first_values[source] is not None
        and requests.last_values[source] is not None
        and requests.last_values[source] >= lower
        and requests.first_values[source] <= upper
    )


def _range_destinations(
    rank: int, summary: RangeInputSummary, requests: RangeRequests
) -> tuple[int, ...]:
    if not summary.has_rows:
        return ()
    first = summary.chunk_bounds[0].first
    last = summary.chunk_bounds[-1].last
    return tuple(
        destination
        for destination in range(len(requests.row_counts))
        if destination != rank
        and requests.lower_values[destination] is not None
        and requests.upper_values[destination] is not None
        and last >= requests.lower_values[destination]
        and first <= requests.upper_values[destination]
    )


async def _range_sends(
    context: Context,
    buffered: BufferedInput,
    policy: RangeOverlapPolicy,
    summary: RangeInputSummary,
    requests: RangeRequests,
    destinations: Sequence[int],
    row_offset: int,
) -> list[GhostSend]:
    sends: list[GhostSend] = []
    records = {
        (chunk.row_start, chunk.row_stop): chunk
        for chunk in buffered.chunks
        if chunk.num_rows
    }
    for bounds in summary.chunk_bounds:
        candidates = [
            destination
            for destination in destinations
            if requests.lower_values[destination] is not None
            and requests.upper_values[destination] is not None
            and bounds.last >= requests.lower_values[destination]
            and bounds.first <= requests.upper_values[destination]
        ]
        if not candidates:
            continue
        record = records[(bounds.row_start, bounds.row_stop)]
        global_start = bounds.row_start + row_offset
        global_stop = bounds.row_stop + row_offset
        edges = []
        for destination in candidates:
            lower = requests.lower_values[destination]
            upper = requests.upper_values[destination]
            assert lower is not None
            assert upper is not None
            whole_start = lower < bounds.first or (
                policy.start_closed and lower == bounds.first
            )
            whole_stop = upper > bounds.last or (
                policy.end_closed and upper == bounds.last
            )
            if whole_start and whole_stop:
                sends.append(GhostSend(destination, global_start, global_stop))
            else:
                edges.append((destination, lower, upper, whole_start, whole_stop))
        if not edges:
            continue
        async with buffered.borrow(record) as table_chunk:
            stream = table_chunk.stream
            index_column = table_chunk.table_view().columns()[policy.index]
            for destination, lower, upper, whole_start, whole_stop in edges:
                if whole_start:
                    start = global_start
                else:
                    start = global_start + await policy._insertion_point(
                        context,
                        index_column,
                        _single_ordering_value_column(
                            lower,
                            policy.index_dtype,
                            stream,
                            context.br(),
                        ),
                        policy.find_start,
                        policy.index_dtype,
                        stream,
                    )
                if whole_stop:
                    stop = global_stop
                else:
                    stop = global_start + await policy._insertion_point(
                        context,
                        index_column,
                        _single_ordering_value_column(
                            upper,
                            policy.index_dtype,
                            stream,
                            context.br(),
                        ),
                        policy.find_end,
                        policy.index_dtype,
                        stream,
                    )
                if start < stop:
                    sends.append(
                        GhostSend(
                            destination,
                            start,
                            stop,
                        )
                    )
    coalesced: list[GhostSend] = []
    for send in sorted(sends, key=lambda item: (item.destination, item.start)):
        if (
            coalesced
            and coalesced[-1].destination == send.destination
            and coalesced[-1].stop == send.start
        ):
            previous = coalesced[-1]
            coalesced[-1] = GhostSend(previous.destination, previous.start, send.stop)
        else:
            coalesced.append(send)
    return coalesced


@define_actor()
async def rolling_actor(
    context: Context,
    comm: Communicator,
    ir: Rolling | FixedSizeRolling,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    *,
    collective_id: int,
) -> None:
    """Streaming actor for rolling operations."""
    async with shutdown_on_error(
        context, chs_in=(ch_in,), chs_out=(ch_out,), trace_ir=ir, ir_context=ir_context
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        metadata_in = await recv_metadata(ch_in, context)
        with RollingManager(context, ir, ir_context) as manager:
            async with RollingInput(
                context,
                comm,
                ch_in,
                manager.policy,
                metadata_in,
                collective_id=collective_id,
            ) as rolling_input:
                await send_metadata(
                    ch_out,
                    context,
                    ChannelMetadata(
                        local_count=metadata_in.local_count,
                        partitioning=maybe_remap_partitioning(
                            ir,
                            metadata_in.partitioning,
                            child_ir=ir.children[0],
                            context=context,
                        ),
                        duplicated=metadata_in.duplicated,
                    ),
                )
                if tracer is not None and metadata_in.duplicated:
                    tracer.set_duplicated()
                execute = manager.execute(ch_out, tracer, rolling_input)
                tasks = rolling_input.producer_tasks()
                if tasks:
                    await gather_in_task_group(*tasks, execute)
                else:
                    await execute


def generate_rolling_sub_network(
    ir: Rolling | FixedSizeRolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for rolling operations."""
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    actors[ir] = [
        rolling_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context_for_node(rec, ir),
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_id=collective_id,
        )
    ]
    return actors, channels


@generate_ir_sub_network.register(FixedSizeRolling)
def _(
    ir: FixedSizeRolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for fixed-size rolling expressions."""
    return generate_rolling_sub_network(ir, rec)


@generate_ir_sub_network.register(Rolling)
def _(
    ir: Rolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for a Rolling operation."""
    if len(ir.keys) > 0 or ir.zlice is not None:
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    return generate_rolling_sub_network(ir, rec)
