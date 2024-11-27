# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition Dask execution."""

from __future__ import annotations

import itertools
import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

import cudf_polars.experimental.io  # noqa: F401
from cudf_polars.dsl.ir import IR, Union
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import PartitionInfo, _concat, get_key_name
from cudf_polars.experimental.dispatch import (
    generate_ir_tasks,
    lower_ir_node,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.experimental.dispatch import LowerIRTransformer


def lower_ir_graph(ir: IR) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR graph and extract partitioning information.

    Parameters
    ----------
    ir
        Root of the graph to rewrite.

    Returns
    -------
    new_ir, partition_info
        The rewritten graph, and a mapping from unique nodes
        in the new graph to associated partitioning information.

    Notes
    -----
    This function traverses the unique nodes of the graph with
    root `ir`, and applies :func:`lower_ir_node` to each node.

    See Also
    --------
    lower_ir_node
    """
    from cudf_polars.dsl.traversal import CachingVisitor

    mapper = CachingVisitor(
        lower_ir_node,
        state={"default_mapper": CachingVisitor(_lower_ir_single)},
    )
    return mapper(ir)


def _lower_ir_single(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Single-partition fall-back for lower_ir_node
    # (Used by rec.state["default_mapper"])
    if len(ir.children) == 0:
        # Default leaf node has single partition
        return ir, {
            ir: PartitionInfo(count=1)
        }  # pragma: no cover; Missed by pylibcudf executor

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=False)
    partition_info = reduce(operator.or_, _partition_info)

    # Check that child partitioning is supported
    if any(partition_info[c].count > 1 for c in children):
        raise NotImplementedError(
            f"Class {type(ir)} does not support multiple partitions."
        )  # pragma: no cover

    # Return reconstructed node and partition-info dict
    partition = PartitionInfo(count=1)
    new_node = ir.reconstruct(children)
    partition_info[new_node] = partition
    return new_node, partition_info


def task_graph(
    ir: IR, partition_info: MutableMapping[IR, PartitionInfo]
) -> tuple[MutableMapping[Any, Any], str | tuple[str, int]]:
    """
    Construct a task graph for evaluation of an IR graph.

    Parameters
    ----------
    ir
        Root of the graph to rewrite.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.

    Returns
    -------
    graph
        A Dask-compatible task graph for the entire
        IR graph with root `ir`.

    Notes
    -----
    This function traverses the unique nodes of the
    graph with root `ir`, and extracts the tasks for
    each node with :func:`generate_ir_tasks`.

    See Also
    --------
    generate_ir_tasks
    """
    graph = reduce(
        operator.or_,
        (generate_ir_tasks(node, partition_info) for node in traversal(ir)),
    )

    key_name = get_key_name(ir)
    partition_count = partition_info[ir].count
    if partition_count > 1:
        graph[key_name] = (_concat, list(partition_info[ir].keys(ir)))
        return graph, key_name
    else:
        return graph, (key_name, 0)


def evaluate_dask(ir: IR) -> DataFrame:
    """Evaluate an IR graph with Dask."""
    from dask import get

    ir, partition_info = lower_ir_graph(ir)

    graph, key = task_graph(ir, partition_info)
    return get(graph, key)


##
## IR
##


@lower_ir_node.register(IR)
def _(ir: IR, rec: LowerIRTransformer) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Single-partition default (see: _lower_ir_single)
    return rec.state["default_mapper"](ir)


@generate_ir_tasks.register(IR)
def _(
    ir: IR, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    # Single-partition default behavior.
    # This is used by `generate_ir_tasks` for all unregistered IR sub-types.
    if partition_info[ir].count > 1:
        raise NotImplementedError(
            f"Failed to generate multiple output tasks for {ir}."
        )  # pragma: no cover

    child_names = []
    for child in ir.children:
        child_names.append(get_key_name(child))
        if partition_info[child].count > 1:
            raise NotImplementedError(
                f"Failed to generate tasks for {ir} with child {child}."
            )  # pragma: no cover

    key_name = get_key_name(ir)
    return {
        (key_name, 0): (
            ir.do_evaluate,
            *ir._non_child_args,
            *((child_name, 0) for child_name in child_names),
        )
    }


##
## Union
##


@lower_ir_node.register(Union)
def _(
    ir: Union, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # zlice must be None
    if ir.zlice is not None:
        return rec.state["default_mapper"](ir)  # pragma: no cover

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Partition count is the sum of all child partitions
    count = sum(partition_info[c].count for c in children)

    # Return reconstructed node and partition-info dict
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=count)
    return new_node, partition_info


@generate_ir_tasks.register(Union)
def _(
    ir: Union, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    key_name = get_key_name(ir)
    partition = itertools.count()
    return {
        (key_name, next(partition)): child_key
        for child in ir.children
        for child_key in partition_info[child].keys(child)
    }
