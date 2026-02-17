# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions and classes for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
import operator
import struct
from contextlib import asynccontextmanager
from functools import reduce
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.tracing import LOG_TRACES, Scope

try:
    import structlog
    import structlog.contextvars
except ImportError:
    pass

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

from cudf_polars.containers import DataFrame
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.spillable_messages import SpillableMessages

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer
    from cudf_polars.typing import DataType


@asynccontextmanager
async def shutdown_on_error(
    context: Context,
    *channels: Channel[Any],
    trace_ir: IR | None = None,
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

    Yields
    ------
    ActorTracer | None
        An actor tracer for collecting stats (if tracing enabled), else None.
    """
    # Create tracer only if LOG_TRACES is enabled and IR is provided
    tracer: ActorTracer | None = None
    if LOG_TRACES and trace_ir is not None:
        from cudf_polars.experimental.rapidsmpf.tracing import (
            ActorTracer,
        )

        ir_id = trace_ir.get_stable_id()
        ir_type = type(trace_ir).__name__
        tracer = ActorTracer(ir_id, ir_type)
        structlog.contextvars.bind_contextvars(actor_ir_id=ir_id, actor_ir_type=ir_type)

    try:
        yield tracer
    except BaseException:
        await asyncio.gather(*(ch.shutdown(context) for ch in channels))
        raise
    finally:
        if tracer is not None:
            log = structlog.get_logger()
            record: dict[str, Any] = {
                "scope": Scope.ACTOR.value,
                "actor_ir_id": tracer.ir_id,
                "actor_ir_type": tracer.ir_type,
                "chunk_count": tracer.chunk_count,
                "duplicated": tracer.duplicated,
            }
            if tracer.row_count is not None:
                record["rows"] = tracer.row_count
            if tracer.decision is not None:
                record["decision"] = tracer.decision
            log.info("Streaming Actor", **record)
            structlog.contextvars.unbind_contextvars("actor_ir_id", "actor_ir_type")


def remap_partitioning(
    partitioning: Partitioning | None,
    old_schema: Mapping[str, DataType],
    new_schema: Mapping[str, DataType],
) -> Partitioning | None:
    """
    Remap partitioning column indices from old schema to new schema.

    Since HashScheme uses column indices rather than names, we need to
    remap indices when propagating partitioning through operations that
    may change the schema (column order or presence).

    Parameters
    ----------
    partitioning
        The partitioning to remap.
    old_schema
        The schema where the partitioning was established.
    new_schema
        The new schema to remap to.

    Returns
    -------
    The remapped partitioning, or None if the inter-rank partitioning
    columns are not present in the new schema.
    """
    if partitioning is None:
        return None

    old_names = list(old_schema.keys())
    new_name_to_idx = {name: i for i, name in enumerate(new_schema.keys())}

    def remap_hash_scheme(hs: HashScheme | None | str) -> HashScheme | None | str:
        if isinstance(hs, HashScheme):
            try:
                new_indices = tuple(
                    new_name_to_idx[old_names[i]] for i in hs.column_indices
                )
            except (IndexError, KeyError):
                return None  # Column missing in old or new schema
            return HashScheme(new_indices, hs.modulus)
        else:
            return hs  # None or "inherit" passes through unchanged

    new_inter_rank = remap_hash_scheme(partitioning.inter_rank)
    new_local = remap_hash_scheme(partitioning.local)
    return Partitioning(inter_rank=new_inter_rank, local=new_local)


def select_preserves_partitioning(
    ir: IR,
    partitioning: Partitioning | None,
    old_schema: Mapping[str, DataType],
    new_schema: Mapping[str, DataType],
) -> Partitioning | None:
    """
    Check if a Select node preserves partitioning and remap if so.

    A Select preserves partitioning if all partition key columns are
    output as simple Col references (unchanged values). Other columns
    can be computed expressions - only the partition keys matter.

    Parameters
    ----------
    ir
        The Select IR node.
    partitioning
        The input partitioning.
    old_schema
        The input schema (child schema).
    new_schema
        The output schema (Select's schema).

    Returns
    -------
    The remapped partitioning if partition keys are preserved, else None.
    """
    from cudf_polars.dsl.expr import Col
    from cudf_polars.dsl.ir import Select

    if partitioning is None:
        return None

    if not isinstance(ir, Select):
        return None

    inter_rank = partitioning.inter_rank
    if inter_rank is None or inter_rank == "inherit":
        # No specific partitioning to preserve
        return remap_partitioning(partitioning, old_schema, new_schema)

    # Get partition key column names from indices
    old_names = list(old_schema.keys())
    try:
        partition_key_names = {old_names[i] for i in inter_rank.column_indices}
    except IndexError:
        return None

    # Build a map from output column name to its expression
    output_expr_map = {ne.name: ne.value for ne in ir.exprs}

    # Check if each partition key column is output as a simple Col reference
    # with the same name (meaning the values are unchanged)
    for key_name in partition_key_names:
        if key_name not in output_expr_map:
            # Partition key column not in output
            return None
        expr = output_expr_map[key_name]
        if not isinstance(expr, Col) or expr.name != key_name:
            # Not a simple Col reference or different source column
            return None

    # All partition keys are preserved, remap the partitioning
    return remap_partitioning(partitioning, old_schema, new_schema)


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
    ir_context: Any,
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
    return TableChunk.from_pylibcudf_table(df.table, chunk.stream, exclusive_view=True)


async def evaluate_chunk(
    context: Context,
    chunk: TableChunk,
    *irs: IR,
    ir_context: Any,
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
                _evaluate_chunk_sync, chunk, single_ir, ir_context
            )
        return chunk


async def concat_batch(
    batch: list[TableChunk],
    context: Context,
    schema: Mapping[str, DataType],
    ir_context: Any,
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
        del batch
    return TableChunk.from_pylibcudf_table(df.table, df.stream, exclusive_view=True)


async def evaluate_batch(
    batch: list[TableChunk],
    context: Context,
    *irs: IR,
    ir_context: Any,
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
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata: ChannelMetadata,
    initial_chunks: list[TableChunk] | None = None,
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
    initial_chunks
        Optional chunks already received (e.g., during sampling) that should
        be forwarded without re-evaluation.
    handle_empty_input
        If True and no chunks are received, create an empty chunk and evaluate
        it. Use for operations like aggregations that always produce output.
    tracer
        Optional tracer for runtime metrics.
    """
    # Send output metadata preserving partitioning
    await send_metadata(ch_out, context, metadata)
    if tracer is not None and metadata.duplicated:
        tracer.set_duplicated()

    seq_num = 0
    received_any = bool(initial_chunks)

    # Forward initial chunks without re-evaluation (if any)
    if initial_chunks:
        for chunk in initial_chunks:
            if tracer is not None:
                tracer.add_chunk(table=chunk.table_view())
            await ch_out.send(context, Message(seq_num, chunk))
            seq_num += 1

    # Process remaining chunks
    while (msg := await ch_in.recv(context)) is not None:
        received_any = True
        result = await evaluate_chunk(
            context, TableChunk.from_message(msg), ir, ir_context=ir_context
        )
        del msg
        if tracer is not None:
            tracer.add_chunk(table=result.table_view())
        await ch_out.send(context, Message(seq_num, result))
        seq_num += 1

    # Handle empty input for aggregation-like operations
    if handle_empty_input and not received_any:
        chunk = empty_table_chunk(ir.children[0], context, ir_context.get_cuda_stream())
        result = await evaluate_chunk(context, chunk, ir, ir_context=ir_context)
        del chunk
        if tracer is not None:
            tracer.add_chunk(table=result.table_view())
        await ch_out.send(context, Message(seq_num, result))

    await ch_out.drain(context)


def is_partitioned_on_keys(
    metadata: ChannelMetadata,
    key_indices: tuple[int, ...],
    nranks: int,
) -> tuple[bool, bool]:
    """
    Check if data is already partitioned on the given keys.

    Parameters
    ----------
    metadata
        The channel metadata.
    key_indices
        The column indices of the keys.
    nranks
        The number of ranks.

    Returns
    -------
    already_partitioned_inter_rank
        Whether the data is already partitioned between ranks.
    already_partitioned_local
        Whether the data is already partitioned within a rank.
    """
    if metadata.partitioning is None:
        return nranks == 1, False

    inter_rank = metadata.partitioning.inter_rank
    local = metadata.partitioning.local
    if nranks > 1 and (
        inter_rank is None
        or inter_rank == "inherit"
        or inter_rank.column_indices != key_indices
    ):
        return False, False

    # Inter-rank is partitioned on keys. Check local.
    if local == "inherit":
        # Local inherits from inter_rank, which is partitioned on keys
        return True, True
    elif local is not None and local.column_indices == key_indices:
        # Local is explicitly partitioned on the same keys
        return True, True
    else:
        # Inter-rank matches but local doesn't
        return True, False


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
    # Create an empty table with the correct schema
    # Use dtype.plc_type to get the full DataType (preserves precision/scale for Decimals)
    empty_columns = [
        plc.column_factories.make_empty_column(dtype.plc_type, stream=stream)
        for dtype in ir.schema.values()
    ]
    empty_table = plc.Table(empty_columns)

    return TableChunk.from_pylibcudf_table(
        empty_table,
        stream,
        exclusive_view=True,
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
    op_id: int,
    *local_values: int,
) -> tuple[int, ...]:
    """
    Allgather local scalar values and sum each across all ranks.

    Parameters
    ----------
    context
        The rapidsmpf context.
    op_id
        The collective operation ID for this allgather.
    *local_values
        One or more local scalar values to contribute.

    Returns
    -------
    tuple[int, ...]
        The sum of each local_value across all ranks.
    """
    n = len(local_values)
    fmt = f"<{'q' * n}"
    data = struct.pack(fmt, *local_values)
    packed = PackedData.from_host_bytes(data, context.br())

    allgather = AllGather(context, op_id)
    allgather.insert(0, packed)
    allgather.insert_finished()

    results = await allgather.extract_all(context, ordered=False)

    totals = [0] * n
    for packed_result in results:
        result_bytes = packed_result.to_host_bytes()
        values = struct.unpack(fmt, result_bytes)
        for i, v in enumerate(values):
            totals[i] += v

    return tuple(totals)
