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
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.table_chunk import TableChunk

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

    __slots__ = ("count", "partitioned_on")
    count: int
    """Chunk-count estimate."""
    partitioned_on: tuple[str, ...]
    """Partitioned-on columns."""

    def __init__(
        self,
        count: int,
        *,
        partitioned_on: tuple[str, ...] = (),
    ):
        self.count = count
        self.partitioned_on = partitioned_on


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
    def create(cls) -> ChannelPair:
        """Create a new ChannelPair with fresh channels."""
        return cls(metadata=Channel(), data=Channel())

    async def send_metadata(self, ctx: Context, metadata: Metadata | None) -> None:
        """
        Send metadata if present, then drain metadata channel.

        Parameters
        ----------
        ctx :
            The streaming context.
        metadata :
            The metadata to send. If None, just drain.
        """
        if metadata is not None:
            msg = Message(0, ArbitraryChunk(metadata))
            await self.metadata.send(ctx, msg)
        await self.metadata.drain(ctx)

    async def recv_metadata(self, ctx: Context) -> Metadata | None:
        """
        Receive metadata from the metadata channel.

        Parameters
        ----------
        ctx :
            The streaming context.

        Returns
        -------
        ChunkMetadata | None
            The metadata, or None if channel is drained.
        """
        msg = await self.metadata.recv(ctx)
        if msg is None:
            return None
        return ArbitraryChunk.from_message(msg).release()


class ChannelManager:
    """A utility class for managing ChannelPair objects."""

    def __init__(self, *, count: int = 1):
        """
        Initialize the ChannelManager with a given number of ChannelPair slots.

        Parameters
        ----------
        count: int
            The number of ChannelPair slots to allocate.
        """
        self._channel_slots = [ChannelPair.create() for _ in range(count)]
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
) -> tuple[list[Any], dict[IR, ChannelManager]]:
    """
    Process children IR nodes and aggregate their nodes and channels.

    This helper function recursively processes all children of an IR node,
    collects their streaming network nodes into a flat list, and merges
    their channel dictionaries.

    Parameters
    ----------
    ir
        The IR node whose children should be processed.
    rec
        Recursive SubNetGenerator callable.

    Returns
    -------
    nodes
        Flat list of all streaming network nodes from all children.
    channels
        Dictionary mapping each child IR node to its ChannelManager.
    """
    if not ir.children:
        return [], {}

    _nodes_list, _channels_list = zip(*(rec(c) for c in ir.children), strict=True)
    nodes: list[Any] = list(reduce(operator.add, _nodes_list, []))
    channels: dict[IR, ChannelManager] = reduce(operator.or_, _channels_list)
    return nodes, channels
