# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Partitioned LogicalPlan nodes."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.expr import NamedExpr
from cudf_polars.dsl.ir import PartitionInfo
from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.nodebase import Node


def get_key_name(node: Node | NamedExpr) -> str:
    """Generate the key name for a Node."""
    if isinstance(node, NamedExpr):
        return f"named-{get_key_name(node.value)}"
    return f"{type(node).__name__.lower()}-{hash(node)}"


@singledispatch
def lower_ir_node(ir: IR, rec) -> IR:
    """Rewrite an IR node with proper partitioning."""
    # Return same node by default
    return ir


def lower_ir_graph(ir: IR) -> IR:
    """Rewrite an IR graph with proper partitioning."""
    from cudf_polars.dsl.traversal import CachingVisitor

    mapper = CachingVisitor(lower_ir_node)
    return mapper(ir)


def _default_ir_parts_info(ir: IR) -> PartitionInfo:
    # Single-partition default behavior.
    # This is used by `ir_parts_info` for
    # all unregistered IR sub-types.
    count = 1
    if ir.children:
        count = max(child.parts.count for child in ir.children)
    if count > 1:
        raise NotImplementedError(
            f"Class {type(ir)} does not support multiple partitions."
        )
    return PartitionInfo(count=count)


@singledispatch
def ir_parts_info(ir: IR) -> PartitionInfo:
    """Return the partitioning info for an IR node."""
    return _default_ir_parts_info(ir)


def _default_ir_tasks(ir: IR) -> MutableMapping[Any, Any]:
    # Single-partition default behavior.
    # This is used by `generate_ir_tasks` for
    # all unregistered IR sub-types.
    if ir.parts.count > 1:
        raise NotImplementedError(f"Failed to generate tasks for {ir}.")

    child_names = []
    for child in ir.children:
        child_names.append(get_key_name(child))
        if child.parts.count > 1:
            raise NotImplementedError(
                f"Failed to generate tasks for {ir} with child {child}."
            )

    key_name = get_key_name(ir)
    return {
        (key_name, 0): (
            ir.do_evaluate,
            *ir._non_child_args,
            *((child_name, 0) for child_name in child_names),
        )
    }


@singledispatch
def generate_ir_tasks(ir: IR) -> MutableMapping[Any, Any]:
    """
    Generate tasks for an IR node.

    An IR node only needs to generate the graph for
    the current IR logic (not including child IRs).
    """
    return _default_ir_tasks(ir)


def task_graph(_ir: IR) -> tuple[MutableMapping[str, Any], str]:
    """Construct a Dask-compatible task graph."""
    ir: IR = lower_ir_graph(_ir)

    graph = {
        k: v
        for layer in [generate_ir_tasks(n) for n in traversal(ir)]
        for k, v in layer.items()
    }
    key_name = get_key_name(ir)
    graph[key_name] = (key_name, 0)

    return graph, key_name


def evaluate_dask(ir: IR) -> DataFrame:
    """Evaluate an IR graph with Dask."""
    from dask import get

    graph, key = task_graph(ir)
    return get(graph, key)
