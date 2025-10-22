# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dispatching for the RapidsMPF streaming engine."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, TypedDict

from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import (
        PartitionInfo,
        StatsCollector,
    )
    from cudf_polars.experimental.rapidsmpf.utils import ChannelManager
    from cudf_polars.utils.config import ConfigOptions


class LowerState(TypedDict):
    """
    State used for lowering an IR node.

    Parameters
    ----------
    config_options
        GPUEngine configuration options.
    stats
        Statistics collector.
    """

    config_options: ConfigOptions
    stats: StatsCollector


LowerIRTransformer: TypeAlias = GenericTransformer[
    "IR", "tuple[IR, MutableMapping[IR, PartitionInfo]]", LowerState
]
"""Protocol for Lowering IR nodes."""


class MulticastInfo(NamedTuple):
    """A named tuple representing multicast information."""

    num_consumers: int
    """The number of consumers."""
    unbounded: bool
    """Whether the node needs unbounded multicast."""


class GenState(TypedDict):
    """
    State used for generating a streaming sub-network.

    Parameters
    ----------
    ctx
        The rapidsmpf context.
    config_options
        GPUEngine configuration options.
    partition_info
        Partition information.
    multicast_nodes
        Dictionary mapping IR nodes to multicast information.
    """

    ctx: Context
    config_options: ConfigOptions
    partition_info: MutableMapping[IR, PartitionInfo]
    multicast_nodes: dict[IR, MulticastInfo]


SubNetGenerator: TypeAlias = GenericTransformer[
    "IR", "tuple[list[Any], dict[IR, ChannelManager]]", GenState
]
"""Protocol for Generating a streaming sub-network."""


@singledispatch
def lower_ir_node(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR node and extract partitioning information.

    Parameters
    ----------
    ir
        IR node to rewrite.
    rec
        Recursive LowerIRTransformer callable.

    Returns
    -------
    new_ir, partition_info
        The rewritten node, and a mapping from unique nodes in
        the full IR graph to associated partitioning information.

    Notes
    -----
    This function is distinct from the `lower_ir_node` function
    in the `parallel` module, because the lowering logic for the
    streaming engine is different for some IR sub-classes.

    See Also
    --------
    lower_ir_graph
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover


@singledispatch
def generate_ir_sub_network(
    ir: IR, rec: SubNetGenerator
) -> tuple[list[Any], dict[IR, ChannelManager]]:
    """
    Generate a sub-network for the RapidsMPF streaming engine.

    Parameters
    ----------
    ir
        IR node to generate tasks for.
    rec
        Recursive SubNetGenerator callable.

    Returns
    -------
    nodes
        List of streaming-network node(s).
    channels
        Dictionary mapping between each IR node and its
        corresponding output ChannelManager object.
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover
