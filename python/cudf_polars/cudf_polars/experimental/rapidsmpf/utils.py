# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions and classes for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
import operator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

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


class Metadata:
    """Metadata payload for an individual ChannelPair."""

    __slots__ = ("count", "duplicated", "partitioned_on")
    count: int
    """Chunk-count estimate."""
    partitioned_on: tuple[str, ...]
    """Partitioned-on columns."""
    duplicated: bool
    """Whether the data is duplicated on all workers."""

    def __init__(
        self,
        count: int,
        *,
        partitioned_on: tuple[str, ...] = (),
        duplicated: bool = False,
    ):
        self.count = count
        self.partitioned_on = partitioned_on
        self.duplicated = duplicated


@dataclass
class ChannelPair:
    """
    A pair of channels for metadata and table data.

    This abstraction ensures that metadata and data are kept separate,
    avoiding ordering issues and making the code more type-safe.

    Attributes
    ----------
    metadata :
        Channel for metadata.
    data :
        Channel for table data chunks.

    Notes
    -----
    This is a placeholder implementation. The metadata channel exists
    but is not used yet. Metadata handling will be fully implemented
    in follow-up work.
    """

    metadata: Channel[ArbitraryChunk]
    data: Channel[TableChunk]

    @classmethod
    def create(cls, context: Context) -> ChannelPair:
        """Create a new ChannelPair with fresh channels."""
        return cls(
            metadata=context.create_channel(),
            data=context.create_channel(),
        )

    async def send_metadata(self, ctx: Context, metadata: Metadata) -> None:
        """
        Send metadata and drain the metadata channel.

        Parameters
        ----------
        ctx :
            The streaming context.
        metadata :
            The metadata to send.
        """
        msg = Message(0, ArbitraryChunk(metadata))
        await self.metadata.send(ctx, msg)
        await self.metadata.drain(ctx)

    async def recv_metadata(self, ctx: Context) -> Metadata:
        """
        Receive metadata from the metadata channel.

        Parameters
        ----------
        ctx :
            The streaming context.

        Returns
        -------
        ChunkMetadata
            The metadata, or None if channel is drained.
        """
        msg = await self.metadata.recv(ctx)
        assert msg is not None, f"Expected Metadata message, got {msg}."
        return ArbitraryChunk.from_message(msg).release()


class ChannelManager:
    """A utility class for managing ChannelPair objects."""

    def __init__(self, context: Context, *, count: int = 1):
        """
        Initialize the ChannelManager with a given number of ChannelPair slots.

        Parameters
        ----------
        context
            The rapidsmpf context.
        count: int
            The number of ChannelPair slots to allocate.
        """
        self._channel_slots = [ChannelPair.create(context) for _ in range(count)]
        self._reserved_output_slots: int = 0
        self._reserved_input_slots: int = 0

    def reserve_input_slot(self) -> ChannelPair:
        """
        Reserve an input channel-pair slot.

        Returns
        -------
        The reserved ChannelPair.
        """
        if self._reserved_input_slots >= len(self._channel_slots):
            raise ValueError("No more input channel-pair slots available")
        pair = self._channel_slots[self._reserved_input_slots]
        self._reserved_input_slots += 1
        return pair

    def reserve_output_slot(self) -> ChannelPair:
        """
        Reserve an output channel-pair slot.

        Returns
        -------
        The reserved ChannelPair.
        """
        if self._reserved_output_slots >= len(self._channel_slots):
            raise ValueError("No more output channel-pair slots available")
        pair = self._channel_slots[self._reserved_output_slots]
        self._reserved_output_slots += 1
        return pair


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
    empty_columns = [
        plc.column_factories.make_empty_column(
            plc.DataType(dtype.id()),
            stream=stream,
        )
        for dtype in ir.schema.values()
    ]
    empty_table = plc.Table(empty_columns)

    return TableChunk.from_pylibcudf_table(
        empty_table,
        stream,
        exclusive_view=True,
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
