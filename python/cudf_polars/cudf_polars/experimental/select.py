# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Select Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.dsl.expressions.aggregation import Agg
from cudf_polars.dsl.expressions.base import NamedExpr
from cudf_polars.dsl.ir import Select
from cudf_polars.dsl.traversal import (
    CachingVisitor,
    reuse_if_unchanged,
    toposort,
    traversal,
)
from cudf_polars.experimental.base import PartitionInfo, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.experimental.expressions import FusedExpr, get_expr_partition_count

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.dsl.expressions.base import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer
    from cudf_polars.typing import ExprTransformer


_SIMPLE_AGGS = ("min", "max", "sum")
_SUPPORTED_AGGS = _SIMPLE_AGGS


def replace_sub_expr(e: Expr, rec: ExprTransformer):
    """Replace a target expression node."""
    mapping = rec.state["mapping"]
    if e in mapping:
        return mapping[e]
    return reuse_if_unchanged(e, rec)


def fuse_expr_graph(expr: Expr) -> FusedExpr:
    """Transform an Expr into a graph of FusedExpr nodes."""
    root = expr
    while True:
        exprs = [
            e
            for e in toposort(root, reverse=True)
            if not (isinstance(e, FusedExpr) or e.is_pointwise)
        ]
        if not exprs:
            if isinstance(root, FusedExpr):
                break  # We are done rewriting root
            exprs = [root]
        old = exprs[0]

        # Check that we can handle old
        if not old.is_pointwise and not (
            isinstance(old, Agg) and old.name in _SUPPORTED_AGGS
        ):
            raise NotImplementedError(
                f"Selection does not support {expr} for multiple partitions."
            )

        # Rewrite root to replace old with FusedExpr(old)
        children = [child for child in traversal([old]) if isinstance(child, FusedExpr)]
        new = FusedExpr(old.dtype, old, *children)
        mapper = CachingVisitor(replace_sub_expr, state={"mapping": {old: new}})
        root = mapper(root)

    return root


def decompose_select(
    ir: Select,
    child: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Decompose a Select expression (if possible)."""
    exprs = [fuse_expr_graph(ne.value) for ne in ir.exprs]

    # TODO: Combine aggregations on distinct columns
    new_node = Select(
        ir.schema,
        [NamedExpr(ir.exprs[i].name, exprs[i]) for i in range(len(exprs))],
        ir.should_broadcast,
        child,
    )

    expr_partition_info = get_expr_partition_count(exprs, partition_info[child].count)
    count = max(expr_partition_info[e] for e in exprs)
    partition_info[new_node] = PartitionInfo(count=count)
    return new_node, partition_info


@lower_ir_node.register(Select)
def _(
    ir: Select, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])
    pi = partition_info[child]
    if pi.count > 1 and not all(
        expr.is_pointwise for expr in traversal([e.value for e in ir.exprs])
    ):
        return decompose_select(ir, child, partition_info)

    new_node = ir.reconstruct([child])
    partition_info[new_node] = pi
    return new_node, partition_info


def evaluate_chunk(
    df: DataFrame,
    expr: Expr,
    children: tuple[Expr, ...],
    *references: Column,
) -> Column:
    """Evaluate a pointwise FusedExpr node."""
    return expr.evaluate(df, mapping=dict(zip(children, references, strict=False)))


def combine_chunks(
    agg: Agg,
    columns: Sequence[Column],
) -> Column:
    """Aggregate a sequence of Columns."""
    return agg.op(
        Column(
            plc.concatenate.concatenate([col.obj for col in columns]),
            name=columns[0].name,
        )
    )


def construct_dataframe(columns: Sequence[Column], names: Sequence[str]) -> DataFrame:
    """Construct a DataFrame from a sequence of Columns."""
    return DataFrame(
        [column.rename(name) for column, name in zip(columns, names, strict=False)]
    )


def make_agg_graph_simple(
    expr: FusedExpr,
    child_name: str,
    expr_partition_count: MutableMapping[Expr, int],
) -> MutableMapping[Any, Any]:
    """Build a simple aggregation graph."""
    agg = expr.sub_expr
    assert isinstance(agg, Agg), f"Expected Agg, got {agg}"
    assert agg.name in _SUPPORTED_AGGS, f"Agg {agg} not supported"

    key_name = get_key_name(expr)
    expr_child_names = [get_key_name(c) for c in expr.children]
    expr_bcast = [expr_partition_count[c] == 1 for c in expr.children]
    input_count = max(expr_partition_count[c] for c in agg.children)

    graph: MutableMapping[Any, Any] = {}

    # Pointwise operations
    chunk_name = f"chunk-{key_name}"
    for i in range(input_count):
        graph[(chunk_name, i)] = (
            evaluate_chunk,
            (child_name, i),
            expr.sub_expr,
            expr.children,
            *[
                (name, 0) if bcast else (name, i)
                for name, bcast in zip(expr_child_names, expr_bcast, strict=False)
            ],
        )

    # Combine results
    graph[(key_name, 0)] = (
        combine_chunks,
        expr.sub_expr,
        list(graph.keys()),
    )

    return graph


def make_pointwise_graph(
    expr: FusedExpr,
    child_name: str,
    expr_partition_count: MutableMapping[Expr, int],
) -> MutableMapping[Any, Any]:
    """Build simple pointwise graph for Select."""
    key_name = get_key_name(expr)
    expr_child_names = [get_key_name(c) for c in expr.children]
    expr_bcast = [expr_partition_count[c] == 1 for c in expr.children]
    count = expr_partition_count[expr]
    return {
        (key_name, i): (
            evaluate_chunk,
            (child_name, i),
            expr.sub_expr,
            expr.children,
            *[
                (name, 0) if bcast else (name, i)
                for name, bcast in zip(expr_child_names, expr_bcast, strict=False)
            ],
        )
        for i in range(count)
    }


def build_select_graph(
    ir: Select, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    """Build complex Select graph."""
    (child,) = ir.children
    child_name = get_key_name(child)
    child_count = partition_info[child].count

    # Build Graph to produce a column for each
    # NamedExpr element of ir.exprs
    graph: MutableMapping[Any, Any] = {}
    expr_partition_count: MutableMapping[Expr, int] = {}
    roots = []
    for ne in ir.exprs:
        assert isinstance(ne.value, FusedExpr), f"{ne.value} is not a FusedExpr"
        expr: FusedExpr = ne.value
        roots.append(expr)
        expr_partition_count = get_expr_partition_count(
            [expr],
            child_count,
            update=expr_partition_count,
        )
        for node in toposort(expr, reverse=True):
            assert isinstance(node, FusedExpr), f"{node} is not a FusedExpr"
            sub_expr = node.sub_expr
            if isinstance(sub_expr, Agg) and sub_expr.name in _SIMPLE_AGGS:
                graph.update(
                    make_agg_graph_simple(node, child_name, expr_partition_count)
                )
            elif node.is_pointwise:
                graph.update(
                    make_pointwise_graph(node, child_name, expr_partition_count)
                )
            else:
                # TODO: Implement "complex" aggs (e.g. mean, std, etc)
                raise NotImplementedError(
                    f"{type(sub_expr)} not supported for multiple partitions."
                )

    # Add task(s) to select the final columns
    name = get_key_name(ir)
    count = max(expr_partition_count[root] for root in roots)
    expr_names = [get_key_name(root) for root in roots]
    expr_bcast = [expr_partition_count[root] == 1 for root in roots]
    for i in range(count):
        graph[(name, i)] = (
            construct_dataframe,
            [
                (name, 0) if bcast else (name, i)
                for name, bcast in zip(expr_names, expr_bcast, strict=False)
            ],
            [ne.name for ne in ir.exprs],
        )

    return graph


@generate_ir_tasks.register(Select)
def _(
    ir: Select, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    fused_exprs = [isinstance(ne.value, FusedExpr) for ne in ir.exprs]
    if any(fused_exprs):
        # Handle FusedExpr-based graph construction
        assert all(fused_exprs), "Partial fusion is not supported"
        return build_select_graph(ir, partition_info)
    else:
        # Simple point-wise graph
        child_name = get_key_name(ir.children[0])
        return {
            key: (
                ir.do_evaluate,
                *ir._non_child_args,
                (child_name, i),
            )
            for i, key in enumerate(partition_info[ir].keys(ir))
        }
