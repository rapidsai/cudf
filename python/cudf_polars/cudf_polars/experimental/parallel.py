# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Partitioned LogicalPlan nodes."""

from __future__ import annotations

import operator
from functools import reduce, singledispatch
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import IR
from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from typing import TypeAlias

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.nodebase import Node
    from cudf_polars.typing import GenericTransformer


class PartitionInfo:
    """
    Partitioning information.

    This class only tracks the partition count (for now).
    """

    __slots__ = ("count",)

    def __init__(self, count: int):
        self.count = count


LowerIRTransformer: TypeAlias = (
    "GenericTransformer[IR, MutableMapping[IR, PartitionInfo]]"
)
"""Protocol for Lowering IR nodes."""


def get_key_name(node: Node) -> str:
    """Generate the key name for a Node."""
    return f"{type(node).__name__.lower()}-{hash(node)}"


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


@lower_ir_node.register(IR)
def _(ir: IR, rec: LowerIRTransformer) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if len(ir.children) == 0:
        # Default leaf node has single partition
        return ir, {ir: PartitionInfo(count=1)}

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=False)
    partition_info = reduce(operator.or_, _partition_info)

    # Check that child partitioning is supported
    count = max(partition_info[c].count for c in children)
    if count > 1:
        raise NotImplementedError(
            f"Class {type(ir)} does not support multiple partitions."
        )  # pragma: no cover

    # Return reconstructed node and partition-info dict
    partition = PartitionInfo(count=1)
    new_node = ir.reconstruct(children)
    partition_info[new_node] = partition
    return new_node, partition_info


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

    mapper = CachingVisitor(lower_ir_node)
    return mapper(ir)


@singledispatch
def generate_ir_tasks(
    ir: IR, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    """
    Generate a task graph for evaluation of an IR node.

    Parameters
    ----------
    ir
        IR node to generate tasks for.
    partition_info
        Partitioning information, obtained from :func:`lower_ir_graph`.

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
    return graph, (get_key_name(ir), 0)


def evaluate_dask(ir: IR) -> DataFrame:
    """Evaluate an IR graph with Dask."""
    from dask import get

    ir, partition_info = lower_ir_graph(ir)

    graph, key = task_graph(ir, partition_info)
    return get(graph, key)
