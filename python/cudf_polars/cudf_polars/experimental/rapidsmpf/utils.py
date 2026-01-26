# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions and classes for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
import operator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

from cudf_polars.containers import DataFrame

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from rapidsmpf.memory.memory_reservation import MemoryReservation
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.spillable_messages import SpillableMessages

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


@asynccontextmanager
async def shutdown_on_error(
    context: Context, *channels: Channel[Any]
) -> AsyncIterator[None]:
    """
    Shutdown on error for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    channels
        The channels to shutdown.
    """
    # TODO: This probably belongs in rapidsmpf.
    try:
        yield
    except BaseException:
        await asyncio.gather(*(ch.shutdown(context) for ch in channels))
        raise


class HashPartitioned:
    """
    Hash-partitioned metadata.

    Attributes
    ----------
    columns
        Columns the data is hash-partitioned on.
    scope
        Whether data is partitioned locally (within a rank) or
        globally (across all ranks).
    count
        The modulus used for hash partitioning (number of partitions).
    """

    __slots__ = ("columns", "count", "scope")

    columns: tuple[str, ...]
    scope: Literal["local", "global"]
    count: int

    def __init__(
        self,
        columns: tuple[str, ...],
        scope: Literal["local", "global"],
        count: int,
    ):
        self.columns = columns
        self.scope = scope
        self.count = count


class Metadata:
    """Metadata payload for an individual ChannelWrapper."""

    __slots__ = (
        "duplicated",
        "global_count",
        "local_count",
        "partitioning",
    )

    # Chunk counts
    local_count: int
    """Local chunk-count estimate for the current rank."""
    global_count: int | None
    """Global chunk-count estimate across all ranks."""

    # Partitioning
    partitioning: HashPartitioned | None
    """How the data is hash-partitioned, or None if not partitioned."""

    # Duplication
    duplicated: bool
    """Whether the data is duplicated (identical) on all workers."""

    def __init__(
        self,
        local_count: int,
        *,
        global_count: int | None = None,
        partitioning: HashPartitioned | None = None,
        duplicated: bool = False,
    ):
        if local_count < 0:  # pragma: no cover
            raise ValueError(f"Local count must be non-negative. Got: {local_count}")
        self.local_count = local_count
        if global_count is not None and global_count < 0:  # pragma: no cover
            raise ValueError(f"Global count must be non-negative. Got: {global_count}")
        self.global_count = global_count
        self.partitioning = partitioning
        self.duplicated = duplicated


@dataclass
class ChannelWrapper:
    """
    A wrapper around a RapidsMPF Channel.

    This abstraction provides convenience methods for sending and receiving
    metadata alongside data, using the channel's native metadata stream.

    Attributes
    ----------
    data :
        The underlying channel for both metadata and table data.
    """

    data: Channel[TableChunk]

    @classmethod
    def create(cls, context: Context) -> ChannelWrapper:
        """Create a new ChannelWrapper with a fresh channel."""
        return cls(data=context.create_channel())

    async def send_metadata(self, ctx: Context, metadata: Metadata) -> None:
        """
        Send metadata and drain the metadata stream.

        Parameters
        ----------
        ctx :
            The streaming context.
        metadata :
            The metadata to send.
        """
        msg = Message(0, ArbitraryChunk(metadata))
        await self.data.send_metadata(ctx, msg)
        await self.data.drain_metadata(ctx)

    async def recv_metadata(self, ctx: Context) -> Metadata:
        """
        Receive metadata from the channel's metadata stream.

        Parameters
        ----------
        ctx :
            The streaming context.

        Returns
        -------
        Metadata
            The received metadata.
        """
        msg = await self.data.recv_metadata(ctx)
        assert msg is not None, f"Expected Metadata message, got {msg}."
        return ArbitraryChunk.from_message(msg).release()


class ChannelManager:
    """A utility class for managing ChannelWrapper objects."""

    def __init__(self, context: Context, *, count: int = 1):
        """
        Initialize the ChannelManager with a given number of ChannelWrapper slots.

        Parameters
        ----------
        context
            The rapidsmpf context.
        count: int
            The number of ChannelWrapper slots to allocate.
        """
        self._channel_slots = [ChannelWrapper.create(context) for _ in range(count)]
        self._reserved_output_slots: int = 0
        self._reserved_input_slots: int = 0

    def reserve_input_slot(self) -> ChannelWrapper:
        """
        Reserve an input channel slot.

        Returns
        -------
        The reserved ChannelWrapper.
        """
        if self._reserved_input_slots >= len(self._channel_slots):
            raise ValueError("No more input channel slots available")
        slot = self._channel_slots[self._reserved_input_slots]
        self._reserved_input_slots += 1
        return slot

    def reserve_output_slot(self) -> ChannelWrapper:
        """
        Reserve an output channel slot.

        Returns
        -------
        The reserved ChannelWrapper.
        """
        if self._reserved_output_slots >= len(self._channel_slots):
            raise ValueError("No more output channel slots available")
        slot = self._channel_slots[self._reserved_output_slots]
        self._reserved_output_slots += 1
        return slot


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


@contextmanager
def opaque_reservation(
    context: Context,
    estimated_bytes: int,
) -> Iterator[MemoryReservation]:
    """
    Reserve memory for opaque allocations.

    Parameters
    ----------
    context
        The RapidsMPF context.
    estimated_bytes
        The estimated number of bytes to reserve.

    Yields
    ------
    The memory reservation.
    """
    yield context.br().reserve_device_memory_and_spill(
        estimated_bytes, allow_overbooking=True
    )
