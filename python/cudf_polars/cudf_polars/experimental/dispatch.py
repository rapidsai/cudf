# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition dispatch functions."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl import ir
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.base import (
        ColumnStats,
        PartitionInfo,
        StatsCollector,
    )
    from cudf_polars.utils.config import ConfigOptions


class State(TypedDict):
    """
    State used for lowering IR nodes.

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
    "ir.IR", "tuple[ir.IR, MutableMapping[ir.IR, PartitionInfo]]", State
]
"""Protocol for Lowering IR nodes."""


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
    This function is used by `lower_ir_graph`.

    See Also
    --------
    lower_ir_graph
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover


@singledispatch
def generate_ir_tasks(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    context: IRExecutionContext,
) -> MutableMapping[Any, Any]:
    """
    Generate a task graph for evaluation of an IR node.

    Parameters
    ----------
    ir
        IR node to generate tasks for.
    partition_info
        Partitioning information, obtained from :func:`lower_ir_graph`.
    context
        Runtime context for IR node execution.

    Returns
    -------
    mapping
        A (partial) dask task graph for the evaluation of an ir node.

    Notes
    -----
    Task generation should only produce the tasks for the current node,
    referring to child tasks by name.

    See Also
    --------
    task_graph
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover


@singledispatch
def initialize_column_stats(
    ir: IR, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    """
    Initialize column statistics for an IR node.

    Parameters
    ----------
    ir
        The IR node to collect source statistics for.
    stats
        The `StatsCollector` object containing known source statistics.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    base_stats_mapping
        Mapping between column names and base ``ColumnStats`` objects.

    Notes
    -----
    Base column stats correspond to ``ColumnStats`` objects **without**
    populated ``unique_stats`` information. The purpose of this function
    is to propagate ``DataSourceInfo`` references and set ``children``
    attributes for each column of each IR node.
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover


@singledispatch
def update_column_stats(
    ir: IR,
    stats: StatsCollector,
    config_options: ConfigOptions,
) -> None:
    """
    Finalize local column statistics for an IR node.

    Parameters
    ----------
    ir
        The IR node to finalize local column statistics for.
    stats
        The `StatsCollector` object containing known statistics.
    config_options
        GPUEngine configuration options.
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover
