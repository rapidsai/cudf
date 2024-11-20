# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Partitioned LogicalPlan nodes."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.expr import NamedExpr
from cudf_polars.dsl.ir import (
    Filter,
    GroupBy,
    HStack,
    Join,
    Projection,
    Scan,
    Select,
    Union,
)
from cudf_polars.dsl.traversal import reuse_if_unchanged, traversal

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.nodebase import Node


class PartitionInfo:
    """
    Partitioning information.

    This class only tracks the partition count (for now).
    """

    __slots__ = ("count",)

    def __init__(self, count: int):
        self.count = count


# The hash of an IR object must always map to a
# unique PartitionInfo object, and we can cache
# this mapping until evaluation is complete.
_IR_PARTS_CACHE: MutableMapping[int, PartitionInfo] = {}


def _clear_parts_info_cache() -> None:
    """Clear cached partitioning information."""
    _IR_PARTS_CACHE.clear()


def get_key_name(node: Node | NamedExpr) -> str:
    """Generate the key name for a Node."""
    if isinstance(node, NamedExpr):
        return f"named-{get_key_name(node.value)}"  # pragma: no cover
    return f"{type(node).__name__.lower()}-{hash(node)}"


@singledispatch
def lower_ir_node(ir: IR, rec) -> IR:
    """Rewrite an IR node with proper partitioning."""
    # Return same node by default
    return reuse_if_unchanged(ir, rec)


def lower_ir_graph(ir: IR) -> IR:
    """Rewrite an IR graph with proper partitioning."""
    from cudf_polars.dsl.traversal import CachingVisitor

    mapper = CachingVisitor(lower_ir_node)
    return mapper(ir)


def _default_ir_parts_info(ir: IR) -> PartitionInfo:
    # Single-partition default behavior.
    # This is used by `_ir_parts_info` for all unregistered IR sub-types.
    count = max((ir_parts_info(child).count for child in ir.children), default=1)
    if count > 1:
        raise NotImplementedError(
            f"Class {type(ir)} does not support multiple partitions."
        )  # pragma: no cover
    return PartitionInfo(count=count)


def _partitionwise_ir_parts_info(ir: IR) -> PartitionInfo:
    # Simple partitionwise behavior.
    count = max((ir_parts_info(child).count for child in ir.children), default=1)
    return PartitionInfo(count=count)


@singledispatch
def _ir_parts_info(ir: IR) -> PartitionInfo:
    """IR partitioning-info dispatch."""
    return _default_ir_parts_info(ir)


def ir_parts_info(ir: IR) -> PartitionInfo:
    """Return the partitioning info for an IR node."""
    key = hash(ir)
    try:
        return _IR_PARTS_CACHE[key]
    except KeyError:
        _IR_PARTS_CACHE[key] = _ir_parts_info(ir)
        return _IR_PARTS_CACHE[key]


def _default_ir_tasks(ir: IR) -> MutableMapping[Any, Any]:
    # Single-partition default behavior.
    # This is used by `generate_ir_tasks` for all unregistered IR sub-types.
    if ir_parts_info(ir).count > 1:
        raise NotImplementedError(
            f"Failed to generate multiple output tasks for {ir}."
        )  # pragma: no cover

    child_names = []
    for child in ir.children:
        child_names.append(get_key_name(child))
        if ir_parts_info(child).count > 1:
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


def _partitionwise_ir_tasks(ir: IR) -> MutableMapping[Any, Any]:
    # Simple partitionwise behavior.
    child_names = []
    counts = []
    for child in ir.children:
        child_names.append(get_key_name(child))
        counts.append(ir_parts_info(child).count)
    counts = counts or [1]
    if len(set(counts)) > 1:
        raise NotImplementedError(
            f"Mismatched partition counts not supported: {counts}"
        )

    key_name = get_key_name(ir)
    return {
        (key_name, i): (
            ir.do_evaluate,
            *ir._non_child_args,
            *((child_name, i) for child_name in child_names),
        )
        for i in range(counts[0])
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
    partition_count = ir_parts_info(ir).count
    if partition_count:
        graph[key_name] = (_concat, [(key_name, i) for i in range(partition_count)])
    else:
        graph[key_name] = (key_name, 0)

    _clear_parts_info_cache()
    return graph, key_name


def evaluate_dask(ir: IR) -> DataFrame:
    """Evaluate an IR graph with Dask."""
    from dask import get

    graph, key = task_graph(ir)
    return get(graph, key)


def _concat(dfs: Sequence[DataFrame]) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return Union.do_evaluate(None, *dfs)


##
## Scan
##


@lower_ir_node.register(Scan)
def _(ir: Scan, rec) -> IR:
    import cudf_polars.experimental.io as _io

    return _io.lower_scan_node(ir, rec)


##
## Select
##


@lower_ir_node.register(Select)
def _(ir: Select, rec) -> IR:
    import cudf_polars.experimental.select as _select

    return _select.lower_select_node(ir, rec)


##
## HStack
##


@_ir_parts_info.register(HStack)
def _(ir: HStack) -> PartitionInfo:
    return _partitionwise_ir_parts_info(ir)


@generate_ir_tasks.register(HStack)
def _(ir: HStack) -> MutableMapping[Any, Any]:
    return _partitionwise_ir_tasks(ir)


##
## Filter
##


## TODO: Can filter expressions include aggregations?


@_ir_parts_info.register(Filter)
def _(ir: Filter) -> PartitionInfo:
    return _partitionwise_ir_parts_info(ir)


@generate_ir_tasks.register(Filter)
def _(ir: Filter) -> MutableMapping[Any, Any]:
    return _partitionwise_ir_tasks(ir)


##
## Projection
##


@_ir_parts_info.register(Projection)
def _(ir: Projection) -> PartitionInfo:
    return _partitionwise_ir_parts_info(ir)


@generate_ir_tasks.register(Projection)
def _(ir: Projection) -> MutableMapping[Any, Any]:
    return _partitionwise_ir_tasks(ir)


##
## GroupBy
##


@lower_ir_node.register(GroupBy)
def _(ir: GroupBy, rec) -> IR:
    import cudf_polars.experimental.groupby as _groupby

    return _groupby.lower_groupby_node(ir, rec)


##
## Join
##


@lower_ir_node.register(Join)
def _(ir: Join, rec) -> IR:
    import cudf_polars.experimental.join as _join

    return _join.lower_join_node(ir, rec)
