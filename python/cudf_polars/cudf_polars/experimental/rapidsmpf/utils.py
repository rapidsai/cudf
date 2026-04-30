# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions and classes for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import operator
import struct
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)

import pylibcudf as plc
import rmm.mr

import cudf_polars.dsl.tracing
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.dsl.ir import Cache, Filter, GroupBy, HStack, Join, Projection, Select
from cudf_polars.dsl.tracing import Scope
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine, Iterator

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.spillable_messages import SpillableMessages

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer
    from cudf_polars.typing import DataType, Schema


@contextlib.contextmanager
def set_memory_resource(mr: rmm.mr.DeviceMemoryResource) -> Iterator[None]:
    """
    Context manager that temporarily sets ``mr`` as the current device resource.

    On entry, ``mr`` is installed via ``rmm.mr.set_current_device_resource(mr)``.
    On exit, the previously active resource is restored unconditionally.

    Parameters
    ----------
    mr
        The memory resource to activate for the duration of the block.
    """
    old = rmm.mr.get_current_device_resource()
    rmm.mr.set_current_device_resource(mr)
    try:
        yield
    finally:
        rmm.mr.set_current_device_resource(old)


async def gather_in_task_group(*coroutines: Coroutine[Any, Any, Any]) -> list[Any]:
    """
    asyncio.gather-like API for running tasks in a asyncio.TaskGroup.

    Parameters
    ----------
    coroutines
        Tasks to execute.

    Returns
    -------
    list[Any]
        The results of the coroutines.
    """
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(coro) for coro in coroutines]
    return [task.result() for task in tasks]


@asynccontextmanager
async def shutdown_on_error(
    context: Context,
    *channels: Channel[Any],
    trace_ir: IR,
    ir_context: IRExecutionContext | None = None,
) -> AsyncIterator[ActorTracer | None]:
    """
    Shutdown on error for rapidsmpf.

    This context manager handles channel cleanup on errors and optionally
    emits structlog tracing events when LOG_TRACES is enabled.

    Parameters
    ----------
    context
        The rapidsmpf context.
    channels
        The channels to shutdown on error.
    trace_ir
        Optional IR node to enable tracing for this streaming actor.
        When provided and LOG_TRACES is enabled, an ActorTracer
        is yielded for collecting stats, and a structlog event is
        emitted on exit.
    ir_context
        The IR execution context from cudf-polars. This is used to propagate
        the query_id to the structlog logs emitted in this context.

    Yields
    ------
    ActorTracer | None
        An actor tracer for collecting stats (if tracing enabled), else None.
    """
    # Create tracer only if LOG_TRACES is enabled and IR is provided
    tracer: ActorTracer | None = None
    contextvars: dict[str, Any] = {}
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer

    ir_id = trace_ir.get_stable_id()
    ir_type = type(trace_ir).__name__
    tracer = ActorTracer(ir_id, ir_type)
    contextvars = {"actor_ir_id": ir_id, "actor_ir_type": ir_type}

    if ir_context is not None:
        contextvars["cudf_polars_query_id"] = str(ir_context.query_id)

    with cudf_polars.dsl.tracing.bound_contextvars(**contextvars):
        start = time.monotonic_ns()
        try:
            yield tracer
        except BaseException:
            await gather_in_task_group(
                *itertools.chain.from_iterable(
                    (ch.shutdown(context), ch.shutdown_metadata(context))
                    for ch in channels
                )
            )
            raise
        finally:
            stop = time.monotonic_ns()
            record: dict[str, Any] = {
                "scope": Scope.ACTOR.value,
            }
            if tracer is not None:
                record.update(
                    {
                        "chunk_count": tracer.chunk_count,
                        "duplicated": tracer.duplicated,
                    }
                )
                if tracer.row_count is not None:
                    record["row_count"] = tracer.row_count
                if tracer.decision is not None:
                    record["decision"] = tracer.decision
            cudf_polars.dsl.tracing.log(
                "Streaming Actor", start=start, stop=stop, **record
            )


def _remap_scheme_select(
    select: Select, scheme: HashScheme | None | str
) -> HashScheme | None | str:
    # We must check if this Select node preserves partitioning
    # before we return a remapped scheme.
    if isinstance(scheme, HashScheme):
        # Mapping from old to new names for "col" selection
        old_to_new_names = {
            ne.value.name: ne.name for ne in select.exprs if isinstance(ne.value, Col)
        }
        old_keys = indices_to_names(scheme.column_indices, select.children[0].schema)
        if set(old_keys).issubset(set(old_to_new_names)):
            new_keys = names_to_indices(
                tuple(old_to_new_names[o] for o in old_keys), select.schema
            )
            return HashScheme(new_keys, scheme.modulus)
        return None
    elif scheme not in (None, "inherit"):  # pragma: no cover
        return None  # Guard against new/unsupported scheme types
    return scheme


def _remap_scheme_simple(
    ir: IR, scheme: HashScheme | None | str, child: IR
) -> HashScheme | None | str:
    # Called when we know the IR node preserves partitioning.
    # Just remap to the new schema if possible.
    if isinstance(scheme, HashScheme):
        old_keys = indices_to_names(scheme.column_indices, child.schema)
        try:
            new_indices = names_to_indices(old_keys, ir.schema)
        except (ValueError, IndexError):
            return None  # Column missing in child or output schema
        return HashScheme(new_indices, scheme.modulus)
    return scheme  # None or "inherit" passes through unchanged


def _hstack_to_select(hstack: HStack) -> Select:
    """Translate HStack to the equivalent Select node."""
    col_map = {ne.name: ne for ne in hstack.columns}
    exprs = tuple(
        col_map[name] if name in col_map else NamedExpr(name, Col(dtype, name))
        for name, dtype in hstack.schema.items()
    )
    return Select(hstack.schema, exprs, hstack.should_broadcast, hstack.children[0])


def maybe_remap_partitioning(
    ir: IR, partitioning: Partitioning | None, *, child_ir: IR | None = None
) -> Partitioning | None:
    """
    Remap partitioning for simple IR nodes.

    Parameters
    ----------
    ir
        The IR node.
    partitioning
        The input partitioning.
    child_ir
        The child IR whose schema the partitioning refers to. When None,
        the first child (ir.children[0]) is used.

    Returns
    -------
    The remapped partitioning. When partition keys are not preserved,
    the corresponding scheme will be set to None. When the original
    partitioning is None, the output will also be None.

    Notes
    -----
    A Select preserves partitioning if all partition key columns are
    output as simple Col references (unchanged values). Other columns
    can be computed expressions - only the partition keys matter.
    """
    if partitioning is None:
        return None  # Nothing to preserve
    if isinstance(ir, (Select, HStack)):
        if isinstance(ir, HStack):
            # HStack is a special case of Select
            ir = _hstack_to_select(ir)
        return Partitioning(
            inter_rank=_remap_scheme_select(ir, partitioning.inter_rank),
            local=_remap_scheme_select(ir, partitioning.local),
        )
    if isinstance(ir, GroupBy):
        return Partitioning(
            inter_rank=_remap_scheme_simple(
                ir, partitioning.inter_rank, ir.children[0]
            ),
            local=_remap_scheme_simple(ir, partitioning.local, ir.children[0]),
        )
    if isinstance(ir, (Cache, Join, Projection, Filter)):
        child = child_ir if child_ir is not None else ir.children[0]
        return Partitioning(
            inter_rank=_remap_scheme_simple(ir, partitioning.inter_rank, child),
            local=_remap_scheme_simple(ir, partitioning.local, child),
        )
    return None


async def send_metadata(
    ch: Channel[TableChunk], ctx: Context, metadata: ChannelMetadata
) -> None:
    """
    Send metadata and drain the metadata queue.

    Parameters
    ----------
    ch :
        The channel to send metadata on.
    ctx :
        The streaming context.
    metadata :
        The metadata to send.

    Notes
    -----
    This function copies the metadata before sending, so the caller
    retains ownership of the original metadata object.
    """
    msg = Message(
        0,
        # Copy metadata before sending since Message consumes the handle.
        # Metadata is small, so copying is cheap.
        ChannelMetadata(
            local_count=metadata.local_count,
            partitioning=metadata.partitioning,
            duplicated=metadata.duplicated,
        ),
    )
    await ch.send_metadata(ctx, msg)
    await ch.drain_metadata(ctx)


async def recv_metadata(ch: Channel[TableChunk], ctx: Context) -> ChannelMetadata:
    """
    Receive metadata from a channel's metadata queue.

    Parameters
    ----------
    ch :
        The channel to receive metadata from.
    ctx :
        The streaming context.

    Returns
    -------
    ChannelMetadata
        The received metadata.
    """
    msg = await ch.recv_metadata(ctx)
    assert msg is not None, f"Expected ChannelMetadata message, got {msg}."
    return ChannelMetadata.from_message(msg)


def _evaluate_chunk_sync(
    chunk: TableChunk,
    ir: IR,
    ir_context: IRExecutionContext,
    br: BufferResource,
) -> TableChunk:
    """
    Apply an IR node's do_evaluate to a table chunk (synchronous).

    This is an internal helper. Use `evaluate_chunk` for the async version
    with memory reservation.

    Parameters
    ----------
    chunk
        The input table chunk (must be available).
    ir
        The IR node to evaluate.
    ir_context
        The IR execution context.
    br
        The buffer resource for lifetime tracking.

    Returns
    -------
    The resulting table chunk after evaluation.
    """
    input_schema = ir.children[0].schema
    names = list(input_schema.keys())
    dtypes = list(input_schema.values())
    df = ir.do_evaluate(
        *ir._non_child_args,
        DataFrame.from_table(chunk.table_view(), names, dtypes, chunk.stream),
        context=ir_context,
    )
    return TableChunk.from_pylibcudf_table(
        df.table, df.stream, exclusive_view=True, br=br
    )


async def evaluate_chunk(
    context: Context,
    chunk: TableChunk,
    *irs: IR,
    ir_context: IRExecutionContext,
) -> TableChunk:
    """
    Make chunk available, reserve memory, and evaluate.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    chunk
        The input table chunk.
    irs
        The IR node(s) to evaluate. Evaluations are chained
        in order within a single memory reservation.
    ir_context
        The IR execution context.

    Returns
    -------
    The resulting table chunk after evaluation.
    """
    assert len(irs) > 0, "Expected at least one IR node"
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=chunk.data_alloc_size(),
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        for single_ir in irs:
            chunk = await asyncio.to_thread(
                _evaluate_chunk_sync, chunk, single_ir, ir_context, context.br()
            )
        return chunk


async def concat_batch(
    batch: list[TableChunk],
    context: Context,
    schema: Schema,
    ir_context: IRExecutionContext,
) -> TableChunk:
    """
    Concatenate a list of table chunks.

    Parameters
    ----------
    batch
        The list of table chunks to concatenate.
    context
        The rapidsmpf context.
    schema
        The schema of the table chunks.
    ir_context
        The IR execution context.

    Returns
    -------
    The table chunk after concatenation.
    """
    batch, extra = await make_table_chunks_available_or_wait(
        context,
        batch,
        reserve_extra=sum(c.data_alloc_size() for c in batch),
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        df = await asyncio.to_thread(
            _concat,
            *[
                DataFrame.from_table(
                    c.table_view(),
                    list(schema.keys()),
                    list(schema.values()),
                    c.stream,
                )
                for c in batch
            ],
            context=ir_context,
        )
        if len(batch) > 1:
            del batch
    return TableChunk.from_pylibcudf_table(
        df.table, df.stream, exclusive_view=True, br=context.br()
    )


async def evaluate_batch(
    batch: list[TableChunk],
    context: Context,
    *irs: IR,
    ir_context: IRExecutionContext,
) -> TableChunk:
    """
    Concatenate a list of table chunks and evaluate the result.

    Parameters
    ----------
    batch
        The list of table chunks to evaluate.
    context
        The rapidsmpf context.
    irs
        The IR node(s) to evaluate. Evaluations are chained
        in order within a single memory reservation.
    ir_context
        The IR execution context.

    Returns
    -------
    The table chunk after evaluation.
    """
    assert len(irs) > 0, "Expected at least one IR node"
    first_ir = irs[0]
    input_schema = first_ir.children[0].schema
    chunk = await concat_batch(batch, context, input_schema, ir_context)
    del batch
    return await evaluate_chunk(context, chunk, *irs, ir_context=ir_context)


async def chunkwise_evaluate(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata: ChannelMetadata,
    *,
    handle_empty_input: bool = False,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Apply IR evaluation chunk-by-chunk, preserving partitioning.

    Use when data is already partitioned on the relevant keys and each
    chunk can be processed independently.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    ir
        The IR node to evaluate.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    metadata
        The channel metadata to forward (partitioning preserved).
    handle_empty_input
        If True and no chunks are received, create an empty chunk and evaluate
        it. Use for operations like aggregations that always produce output.
    tracer
        Optional tracer for runtime metrics.
    """
    await send_metadata(ch_out, context, metadata)
    if tracer is not None and metadata.duplicated:
        tracer.set_duplicated()

    received_any = False
    while (msg := await ch_in.recv(context)) is not None:
        received_any = True
        cd = msg.get_content_description()
        seq_num = msg.sequence_number
        with cudf_polars.dsl.tracing.bound_contextvars(
            content_sizes=cd.content_sizes,
            spillable=cd.spillable,
            sequence_number=msg.sequence_number,
        ):
            result = await evaluate_chunk(
                context,
                TableChunk.from_message(msg, br=context.br()),
                ir,
                ir_context=ir_context,
            )
        del msg, cd
        if tracer is not None:
            tracer.add_chunk(table=result.table_view())
        await ch_out.send(context, Message(seq_num, result))

    if handle_empty_input and not received_any:
        chunk = empty_table_chunk(ir.children[0], context, ir_context.get_cuda_stream())
        result = await evaluate_chunk(context, chunk, ir, ir_context=ir_context)
        del chunk
        if tracer is not None:
            tracer.add_chunk(table=result.table_view())
        await ch_out.send(context, Message(0, result))

    await ch_out.drain(context)


def indices_to_names(indices: tuple[int, ...], schema: Schema) -> tuple[str, ...]:
    """
    Return column names for the given column indices in schema order.

    Parameters
    ----------
    indices
        The indices to get names for.
    schema
        The schema to get names from.

    Returns
    -------
    The column names for each index in schema order.
    """
    keys = list(schema.keys())
    return tuple(keys[i] for i in indices)


def names_to_indices(
    names: tuple[str | NamedExpr, ...], schema: Schema
) -> tuple[int, ...]:
    """
    Return column indices for the given names in schema order.

    Accepts either column names (str) or NamedExpr, so it can be used with
    e.g. ir.left_on, ir.right_on as well as plain name tuples.

    Parameters
    ----------
    names
        The names to get indices for.
    schema
        The schema to get indices from.

    Returns
    -------
    The column indices for each name in schema order.
    """
    keys = list(schema.keys())
    str_names = [n.name if isinstance(n, NamedExpr) else n for n in names]
    return tuple(keys.index(n) for n in str_names)


async def replay_buffered_channel(
    context: Context,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    buffered_chunks: dict[int, TableChunk],
    metadata: ChannelMetadata,
    *,
    trace_ir: IR,
) -> None:
    """
    Replay a buffered input channel into an output channel.

    Parameters
    ----------
    context
        The context.
    ch_out
        The new output channel.
    ch_in
        The buffered input channel.
    buffered_chunks
        The buffered chunks to yield first.
    metadata
        The metadata to send to the output channel.
    trace_ir
        The IR node to trace. Passed through to shutdown_on_error.
    """
    async with shutdown_on_error(context, ch_out, ch_in, trace_ir=trace_ir):
        await send_metadata(ch_out, context, metadata)
        for seq_num, chunk in buffered_chunks.items():
            await ch_out.send(context, Message(seq_num, chunk))
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
        await ch_out.drain(context)


@dataclass(frozen=True)
class NormalizedPartitioning:
    """
    Normalized view of channel partitioning for a set of key column indices.

    inter_rank_scheme is None when the channel metadata has no inter-rank
    partitioning on the requested keys. local_scheme is None when the local
    partitioning scheme does not cover the requested keys, and the string
    "inherit" when local layout follows from inter-rank.
    """

    inter_rank_scheme: HashScheme | None
    local_scheme: HashScheme | None | Literal["inherit"]

    def __bool__(self) -> bool:
        """True when both inter-rank and local schemes are present."""
        return self.inter_rank_scheme is not None and self.local_scheme is not None

    def __eq__(self, other: object) -> bool:
        """True when both schemes are equal."""
        if not isinstance(other, NormalizedPartitioning):
            return NotImplemented
        return (
            self.inter_rank_scheme == other.inter_rank_scheme
            and self.local_scheme == other.local_scheme
        )

    def is_aligned_with(self, other: NormalizedPartitioning) -> bool:
        """True when both sides share a compatible hash layout for a chunkwise operation."""
        if not bool(self) or not bool(other):
            return False  # Neither side is "properly" partitioned

        # Check inter-rank alignment
        lhs_inter = self.inter_rank_scheme
        rhs_inter = other.inter_rank_scheme
        if not isinstance(lhs_inter, HashScheme) or not isinstance(
            rhs_inter, HashScheme
        ):
            # TODO: Support OrderScheme when implemented
            return False  # Not hash partitioned
        if lhs_inter.modulus != rhs_inter.modulus:
            return False  # Different moduli
        if len(lhs_inter.column_indices) != len(rhs_inter.column_indices):
            return False  # Different key-column counts

        # Check local alignment
        lhs_loc = self.local_scheme
        rhs_loc = other.local_scheme
        if isinstance(lhs_loc, HashScheme) and isinstance(rhs_loc, HashScheme):
            return lhs_loc.modulus == rhs_loc.modulus and len(
                lhs_loc.column_indices
            ) == len(rhs_loc.column_indices)
        return lhs_loc == rhs_loc == "inherit"

    @classmethod
    def from_keys(
        cls,
        partitioning_metadata: Partitioning | None,
        nranks: int,
        *,
        indices: tuple[int, ...],
        allow_subset: bool = True,
    ) -> NormalizedPartitioning:
        """
        Resolve partitioning from channel metadata and key column indices.

        Parameters
        ----------
        partitioning_metadata
            The channel partitioning metadata.
        nranks
            Number of ranks.
        indices
            Key column indices for the operation.
        allow_subset
            If True, the metadata keys may be a prefix of indices.

        Returns
        -------
        NormalizedPartitioning
            The resolved inter-rank and local partitioning schemes.
        """
        inter_rank_scheme, local_scheme = NormalizedPartitioning._normalize_schemes(
            partitioning_metadata, indices, nranks, allow_subset=allow_subset
        )
        return cls(inter_rank_scheme=inter_rank_scheme, local_scheme=local_scheme)

    @staticmethod
    def _normalize_schemes(
        partitioning_metadata: Partitioning | None,
        key_indices: tuple[int, ...],
        nranks: int,
        *,
        allow_subset: bool = False,
    ) -> tuple[HashScheme | None, HashScheme | None | Literal["inherit"]]:
        """Translate Partitioning metadata into normalized partitioning schemes."""
        trivial = HashScheme(key_indices, 1) if nranks == 1 else None
        if partitioning_metadata is None:
            return trivial, None

        def _keys_match(scheme: object) -> bool:
            if not isinstance(scheme, HashScheme):
                return False
            current = scheme.column_indices
            target = key_indices[: len(current)] if allow_subset else key_indices
            return target == current

        inter_rank = partitioning_metadata.inter_rank
        strict_inter_rank = inter_rank if _keys_match(inter_rank) else None
        inter_rank_scheme: HashScheme | None = strict_inter_rank or trivial
        if inter_rank_scheme is None:
            # Partitioning is meaningless without inter-rank partitioning
            return None, None

        local = partitioning_metadata.local
        local_scheme: HashScheme | None | Literal["inherit"]
        if _keys_match(local):
            assert isinstance(local, HashScheme)
            local_scheme = local
        elif local == "inherit":
            local_scheme = "inherit"
        else:
            local_scheme = None

        # Single-rank normalization: When there is no real inter-rank
        # scheme, local and inter-rank partitioning are equivalent
        if strict_inter_rank is None and trivial is not None:
            if local_scheme not in (None, "inherit"):
                # Translate: (trivial, Scheme) → (Scheme, "inherit")
                return local_scheme, "inherit"
            return trivial, None

        return inter_rank_scheme, local_scheme


class ChannelManager:
    """A utility class for managing Channel objects."""

    def __init__(self, context: Context, *, count: int = 1):
        """
        Initialize the ChannelManager with a given number of channel slots.

        Parameters
        ----------
        context
            The rapidsmpf context.
        count: int
            The number of channel slots to allocate.
        """
        self._channel_slots: list[Channel[TableChunk]] = [
            context.create_channel() for _ in range(count)
        ]
        self._reserved_output_slots: int = 0
        self._reserved_input_slots: int = 0

    def reserve_input_slot(self) -> Channel[TableChunk]:
        """
        Reserve an input channel slot.

        Returns
        -------
        The reserved Channel.
        """
        if self._reserved_input_slots >= len(self._channel_slots):
            raise ValueError("No more input channel slots available")
        self._reserved_input_slots += 1
        return self._channel_slots[self._reserved_input_slots - 1]

    def reserve_output_slot(self) -> Channel[TableChunk]:
        """
        Reserve an output channel slot.

        Returns
        -------
        The reserved Channel.
        """
        if self._reserved_output_slots >= len(self._channel_slots):
            raise ValueError("No more output channel slots available")
        self._reserved_output_slots += 1
        return self._channel_slots[self._reserved_output_slots - 1]


def process_children(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """
    Process children IR nodes and aggregate their nodes and channels.

    This helper function recursively processes all children of an IR node,
    collects their streaming network nodes into a dictionary mapping IR nodes
    to their associated nodes, and merges their channel dictionaries.

    Parameters
    ----------
    ir
        The IR node whose children should be processed.
    rec
        Recursive SubNetGenerator callable.

    Returns
    -------
    nodes
        Dictionary mapping each IR node to its list of streaming network nodes.
    channels
        Dictionary mapping each child IR node to its ChannelManager.
    """
    if not ir.children:
        return {}, {}

    _nodes_list, _channels_list = zip(*(rec(c) for c in ir.children), strict=True)
    nodes: dict[IR, list[Any]] = reduce(operator.or_, _nodes_list)
    channels: dict[IR, ChannelManager] = reduce(operator.or_, _channels_list)
    return nodes, channels


def _make_empty_column(dtype: DataType, stream: Stream) -> plc.Column:
    """
    Create an empty (0-row) column, including for nested types.

    ``plc.column_factories.make_empty_column`` rejects LIST and STRUCT,
    so we build those by hand with the correct child structure.

    Parameters
    ----------
    dtype
        The cudf-polars DataType (carries child-type metadata for nested types).
    stream
        CUDA stream for any device allocations.
    """
    if dtype.id() == plc.TypeId.LIST:
        offsets = plc.Column.from_scalar(
            plc.Scalar.from_py(0, plc.DataType(plc.TypeId.INT32), stream=stream),
            1,
            stream=stream,
        )
        child = _make_empty_column(dtype.children[0], stream)
        return plc.Column(dtype.plc_type, 0, None, None, 0, 0, [offsets, child])

    if dtype.id() == plc.TypeId.STRUCT:
        children = [
            _make_empty_column(child_dtype, stream) for child_dtype in dtype.children
        ]
        return plc.Column(dtype.plc_type, 0, None, None, 0, 0, children)

    return plc.column_factories.make_empty_column(dtype.plc_type, stream=stream)


def empty_table_chunk(ir: IR, context: Context, stream: Stream) -> TableChunk:
    """
    Make an empty table chunk.

    Parameters
    ----------
    ir
        The IR node to use for the schema.
    context
        The rapidsmpf context.
    stream
        The stream to use for the table chunk.

    Returns
    -------
    The empty table chunk.
    """
    empty_columns = [_make_empty_column(dtype, stream) for dtype in ir.schema.values()]
    empty_table = plc.Table(empty_columns)

    return TableChunk.from_pylibcudf_table(
        empty_table,
        stream,
        exclusive_view=True,
        br=context.br(),
    )


def chunk_to_frame(chunk: TableChunk, ir: IR) -> DataFrame:
    """
    Convert a TableChunk to a DataFrame.

    Parameters
    ----------
    chunk
        The TableChunk to convert.
    ir
        The IR node to use for the schema.

    Returns
    -------
    A DataFrame.
    """
    return DataFrame.from_table(
        chunk.table_view(),
        list(ir.schema.keys()),
        list(ir.schema.values()),
        chunk.stream,
    )


def _is_already_partitioned(
    metadata: ChannelMetadata,
    columns_to_hash: tuple[int, ...],
    num_partitions: int,
    nranks: int,
) -> bool:
    """
    Check if data is already hash-partitioned for a shuffle actor.

    Returns True only when the channel carries a HashScheme on the requested
    keys with matching modulus and ``local="inherit"`` — the canonical layout
    produced by a prior global shuffle.
    """
    partitioning = NormalizedPartitioning.from_keys(
        metadata.partitioning,
        nranks,
        indices=columns_to_hash,
        allow_subset=False,
    )
    return (
        isinstance(partitioning.inter_rank_scheme, HashScheme)
        and partitioning.local_scheme == "inherit"
        and partitioning.inter_rank_scheme.modulus == num_partitions
    )


def make_spill_function(
    spillable_messages_list: list[SpillableMessages],
    context: Context,
) -> Callable[[int], int]:
    """
    Create a spill function for a list of SpillableMessages containers.

    This utility creates a spill function that can be registered with a
    SpillManager. The spill function uses a smart spilling strategy that
    prioritizes:
    1. Longest queues first (slow consumers that won't need data soon)
    2. Newest messages first (just arrived, won't be consumed soon)

    This strategy keeps "hot" data (about to be consumed) in fast memory
    while spilling "cold" data (won't be needed for a while) to slower tiers.

    Parameters
    ----------
    spillable_messages_list
        List of SpillableMessages containers to create a spill function for.
    context
        The RapidsMPF context to use for accessing the BufferResource.

    Returns
    -------
    A spill function that takes an amount (in bytes) and returns the
    actual amount spilled (in bytes).

    Notes
    -----
    The spilling strategy is particularly effective for fanout scenarios
    where different consumers may process messages at different rates. By
    prioritizing longest queues and newest messages, we maximize the time
    data can remain in slower memory before it's needed.
    """

    def spill_func(amount: int) -> int:
        """Spill messages from the buffers to free device/host memory."""
        spilled = 0

        # Collect all messages with metadata for smart spilling
        # Format: (message_id, container_idx, queue_length, sm)
        all_messages: list[tuple[int, int, int, SpillableMessages]] = []
        for container_idx, sm in enumerate(spillable_messages_list):
            content_descriptions = sm.get_content_descriptions()
            queue_length = len(content_descriptions)
            all_messages.extend(
                (message_id, container_idx, queue_length, sm)
                for message_id in content_descriptions
            )

        # Spill newest messages first from the longest queues
        # Sort by: (1) queue length descending, (2) message_id descending
        # This prioritizes:
        # - Longest queues (slow consumers that won't need data soon)
        # - Newest messages (just arrived, won't be consumed soon)
        all_messages.sort(key=lambda x: (-x[2], -x[0]))

        # Spill messages until we've freed enough memory
        for message_id, _, _, sm in all_messages:
            if spilled >= amount:
                break
            # Try to spill this message
            spilled += sm.spill(mid=message_id, br=context.br())

        return spilled

    return spill_func


async def allgather_reduce(
    context: Context,
    comm: Communicator,
    op_id: int,
    *local_values: int,
) -> tuple[int, ...]:
    """
    Allgather local scalar values and sum each across all ranks.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    op_id
        The collective operation ID for this allgather.
    *local_values
        One or more local scalar values to contribute.

    Returns
    -------
    tuple[int, ...]
        The sum of each local_value across all ranks.
    """
    if comm.nranks == 1:
        return tuple(local_values)

    n = len(local_values)
    fmt = f"<{'q' * n}"
    data = struct.pack(fmt, *local_values)
    packed = PackedData.from_host_bytes(data, context.br())

    allgather = AllGather(context, comm, op_id)
    try:
        allgather.insert(0, packed)
    finally:
        allgather.insert_finished()

    results = await allgather.extract_all(context, ordered=False)

    totals = [0] * n
    for packed_result in results:
        result_bytes = packed_result.to_host_bytes()
        values = struct.unpack(fmt, result_bytes)
        for i, v in enumerate(values):
            totals[i] += v

    return tuple(totals)
