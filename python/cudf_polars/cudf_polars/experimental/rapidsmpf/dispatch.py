# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch functions for the RAPIDS-MPF streaming engine."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import (
        PartitionInfo,
        StatsCollector,
    )
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
    """

    ctx: Context
    config_options: ConfigOptions
    partition_info: MutableMapping[IR, PartitionInfo]


SubNetGenerator: TypeAlias = GenericTransformer[
    "IR", "tuple[dict[IR, list[Any]], dict[IR, Any]]", GenState
]
"""Protocol for Generating a streaming sub-network."""


@singledispatch
def lower_ir_node(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR node and extract partitioning information for RAPIDS-MPF.

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
    This function is used by `lower_ir_graph_rapidsmpf`.

    See Also
    --------
    lower_ir_graph_rapidsmpf
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover


@singledispatch
def generate_ir_sub_network(
    ir: IR, partition_info: MutableMapping[IR, PartitionInfo]
) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    """
    Generate a sub-network for evaluation of an IR node with rapidsmpf.

    Parameters
    ----------
    ir
        IR node to generate tasks for.
    partition_info
        Partitioning information, obtained from :func:`lower_ir_graph_rapidsmpf`.

    Returns
    -------
    nodes
        Dictionary mapping between each IR node and its
        corresponding streaming-network node(s).
    channels
        Dictionary mapping between each IR node and its
        corresponding streaming-network output channel.
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover
