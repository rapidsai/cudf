# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Partitioned LogicalPlan nodes."""

from __future__ import annotations

import operator
from functools import reduce, singledispatch
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import (
    IR,
    Filter,
    GroupBy,
    HStack,
    Join,
    Projection,
    Scan,
    Select,
    Union,
)
from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence
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
    """Rewrite an IR node with proper partitioning."""
    raise AssertionError(f"Unhandled type {type(ir)}")


def _lower_children(
    ir: IR, rec: LowerIRTransformer
) -> tuple[tuple[IR], MutableMapping[IR, PartitionInfo]]:
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=False)
    partition_info: MutableMapping[IR, PartitionInfo] = reduce(
        operator.or_, _partition_info
    )
    return children, partition_info


@lower_ir_node.register(IR)
def _default_lower_ir_node(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if len(ir.children) == 0:
        # Default leaf node has single partition
        return ir, {ir: PartitionInfo(count=1)}

    # Lower children
    children, partition_info = _lower_children(ir, rec)

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


def _lower_ir_node_partitionwise(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Simple partitionwise behavior
    children, partition_info = _lower_children(ir, rec)
    partition = PartitionInfo(count=max(partition_info[c].count for c in children))
    new_node = ir.reconstruct(children)
    partition_info[new_node] = partition
    return new_node, partition_info


def lower_ir_graph(ir: IR) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Rewrite an IR graph with proper partitioning."""
    from cudf_polars.dsl.traversal import CachingVisitor

    mapper = CachingVisitor(lower_ir_node)
    return mapper(ir)


@singledispatch
def generate_ir_tasks(
    ir: IR, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    """
    Generate tasks for an IR node.

    An IR node only needs to generate the graph for
    the current IR logic (not including child IRs).
    """
    raise AssertionError(f"Unhandled type {type(ir)}")


@generate_ir_tasks.register(IR)
def _default_ir_tasks(
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


def _partitionwise_ir_tasks(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
) -> MutableMapping[Any, Any]:
    # Simple partitionwise behavior.
    child_names = []
    counts = []
    for child in ir.children:
        child_names.append(get_key_name(child))
        counts.append(partition_info[child].count)
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


def task_graph(
    ir: IR, partition_info: MutableMapping[IR, PartitionInfo]
) -> tuple[MutableMapping[str, Any], str]:
    """Construct a Dask-compatible task graph."""
    graph = reduce(
        operator.or_,
        [generate_ir_tasks(node, partition_info) for node in traversal(ir)],
    )

    key_name = get_key_name(ir)
    partition_count = partition_info[ir].count
    if partition_count:
        graph[key_name] = (_concat, [(key_name, i) for i in range(partition_count)])
    else:
        graph[key_name] = (key_name, 0)

    return graph, key_name


def evaluate_dask(ir: IR) -> DataFrame:
    """Evaluate an IR graph with Dask."""
    from dask import get

    ir, partition_info = lower_ir_graph(ir)

    graph, key = task_graph(ir, partition_info)
    return get(graph, key)


def _concat(dfs: Sequence[DataFrame]) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return Union.do_evaluate(None, *dfs)


##
## Scan
##


@lower_ir_node.register(Scan)
def _(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    import cudf_polars.experimental.io as _io

    return _io.lower_scan_node(ir, rec)


##
## Select
##


@lower_ir_node.register(Select)
def _(
    ir: Select, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    import cudf_polars.experimental.select as _select

    return _select.lower_select_node(ir, rec)


##
## HStack
##


@lower_ir_node.register(HStack)
def _(
    ir: HStack, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    return _lower_ir_node_partitionwise(ir, rec)


@generate_ir_tasks.register(HStack)
def _(
    ir: HStack, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    return _partitionwise_ir_tasks(ir, partition_info)


##
## Filter
##


## TODO: Can filter expressions include aggregations?


@lower_ir_node.register(Filter)
def _(
    ir: Filter, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    return _lower_ir_node_partitionwise(ir, rec)


@generate_ir_tasks.register(Filter)
def _(
    ir: Filter, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    return _partitionwise_ir_tasks(ir, partition_info)


##
## Projection
##


@lower_ir_node.register(Projection)
def _(
    ir: Projection, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    return _lower_ir_node_partitionwise(ir, rec)


@generate_ir_tasks.register(Projection)
def _(
    ir: Projection, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    return _partitionwise_ir_tasks(ir, partition_info)


##
## GroupBy
##


@lower_ir_node.register(GroupBy)
def _(
    ir: GroupBy, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    import cudf_polars.experimental.groupby as _groupby

    return _groupby.lower_groupby_node(ir, rec)


##
## Join
##


@lower_ir_node.register(Join)
def _(
    ir: Join, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    import cudf_polars.experimental.join as _join

    return _join.lower_join_node(ir, rec)
