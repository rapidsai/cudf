# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime bindings for optional join prefilters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cudf_polars.streaming.filter_hint import JoinWithPrefilter

if TYPE_CHECKING:
    from typing import Any

    from cudf_streaming.channel_metadata import ChannelMetadata
    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.streaming.core.channel import Channel

    from cudf_polars.dsl.ir import IR, Join
    from cudf_polars.streaming.actor_graph.prefilter import PrefilterDecision
    from cudf_polars.streaming.actor_graph.utils import TableSizeStats
    from cudf_polars.streaming.filter_hint import Prefilter


@dataclass(slots=True)
class JoinInputBinding:
    """Concrete runtime resources for one input to a dynamic join."""

    node: IR
    channel: Channel[TableChunk]
    metadata: ChannelMetadata
    sample: TableSizeStats | None = None


@dataclass(slots=True)
class BoundPrefilter:
    """A logical prefilter bound to its concrete runtime inputs."""

    prefilter: Prefilter
    target: JoinInputBinding
    domain: JoinInputBinding
    cardinality_tag: int
    decision: PrefilterDecision | None = None
    key_channel: Channel[TableChunk] | None = None
    trace: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class JoinBindings:
    """Concrete runtime inputs and optional prefilters for a dynamic join."""

    left: JoinInputBinding
    right: JoinInputBinding
    prefilters: tuple[BoundPrefilter, ...] = ()


def bind_join_inputs(
    ir: Join,
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    cardinality_tags: tuple[int, ...],
) -> JoinBindings:
    """Bind logical join inputs and prefilters to their runtime resources."""
    left = JoinInputBinding(ir.children[0], ch_left, left_metadata)
    right = JoinInputBinding(ir.children[1], ch_right, right_metadata)
    if not isinstance(ir, JoinWithPrefilter):
        return JoinBindings(left, right)

    if len(cardinality_tags) < len(ir.prefilters):
        raise ValueError("Each join prefilter requires a cardinality collective ID")

    sides = {"left": left, "right": right}
    cardinality_tags_iter = iter(cardinality_tags)
    prefilters = []
    for prefilter in ir.prefilters:
        target = sides[prefilter.target_side]
        domain = sides[prefilter.domain_side]
        prefilters.append(
            BoundPrefilter(
                prefilter,
                target,
                domain,
                cardinality_tag=next(cardinality_tags_iter),
            )
        )
    return JoinBindings(left, right, tuple(prefilters))
