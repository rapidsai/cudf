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


class StateInfo:
    """Bag of arbitrary state information."""

    def __init__(self, *, parts_info: MutableMapping[IR, PartitionInfo] | None = None):
        self.__parts_info = parts_info or {}

    def parts(self, ir: IR) -> PartitionInfo:
        """Return partitioning information for an IR node."""
        return self.__parts_info[ir]


def get_key_name(node: Node) -> str:
    """Generate the key name for a Node."""
    return f"{type(node).__name__.lower()}-{hash(node)}"


@singledispatch
def lower_ir_node(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Rewrite an IR node with proper partitioning."""
    raise AssertionError(f"Unhandled type {type(ir)}")


@lower_ir_node.register(IR)
def _default_lower_ir_node(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
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

    # Return reconstructed node and
    partition = PartitionInfo(count=1)
    new_node = ir.reconstruct(children)
    partition_info[new_node] = partition
    return new_node, partition_info


def lower_ir_graph(ir: IR) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Rewrite an IR graph with proper partitioning."""
    from cudf_polars.dsl.traversal import CachingVisitor

    mapper = CachingVisitor(lower_ir_node)
    return mapper(ir)


@singledispatch
def generate_ir_tasks(ir: IR, state: StateInfo) -> MutableMapping[Any, Any]:
    """
    Generate tasks for an IR node.

    An IR node only needs to generate the graph for
    the current IR logic (not including child IRs).
    """
    raise AssertionError(f"Unhandled type {type(ir)}")


@generate_ir_tasks.register(IR)
def _default_ir_tasks(ir: IR, state: StateInfo) -> MutableMapping[Any, Any]:
    # Single-partition default behavior.
    # This is used by `generate_ir_tasks` for all unregistered IR sub-types.
    if state.parts(ir).count > 1:
        raise NotImplementedError(
            f"Failed to generate multiple output tasks for {ir}."
        )  # pragma: no cover

    child_names = []
    for child in ir.children:
        child_names.append(get_key_name(child))
        if state.parts(child).count > 1:
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


def task_graph(ir: IR, state: StateInfo) -> tuple[MutableMapping[str, Any], str]:
    """Construct a Dask-compatible task graph."""
    graph = reduce(
        operator.or_,
        [generate_ir_tasks(node, state) for node in traversal(ir)],
    )
    key_name = get_key_name(ir)
    graph[key_name] = (key_name, 0)

    return graph, key_name


def evaluate_dask(ir: IR) -> DataFrame:
    """Evaluate an IR graph with Dask."""
    from dask import get

    ir, parts_info = lower_ir_graph(ir)

    graph, key = task_graph(ir, StateInfo(parts_info=parts_info))
    return get(graph, key)
