# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ChannelPair abstraction for metadata + data channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from rapidsmpf.streaming.core.channel import Channel

if TYPE_CHECKING:
    from rapidsmpf.streaming.cudf.table_chunk import TableChunk


# Type alias for metadata payloads (placeholder - not used yet)
MetadataPayload: TypeAlias = Any


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

    metadata: Channel[MetadataPayload]
    data: Channel[TableChunk]

    @classmethod
    def create(cls) -> ChannelPair:
        """Create a new ChannelPair with fresh channels."""
        return cls(metadata=Channel(), data=Channel())
