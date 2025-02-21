# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Multi-partition Expr classes and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.aggregation import Agg
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.traversal import (
    CachingVisitor,
    reuse_if_unchanged,
    traversal,
)
from cudf_polars.experimental.base import get_key_name

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expressions.base import AggInfo
    from cudf_polars.typing import ExprTransformer


_SUPPORTED_AGGS = ("count", "min", "max", "sum", "mean")


class FusedExpr(Expr):
    """
    A single Expr node representing a fused Expr sub-graph.

    Notes
    -----
    A FusedExpr may not contain more than one non-pointwise
    Expr nodes. If the object does contain a non-pointwise
    node, that node must be the root of sub_expr.
    """

    __slots__ = ("sub_expr",)
    _non_child = ("dtype", "sub_expr")

    def __init__(
        self,
        dtype: plc.DataType,
        sub_expr: Expr,
        *children: Expr,
    ):
        self.dtype = dtype
        self.sub_expr = sub_expr
        self.children = children
        self.is_pointwise = sub_expr.is_pointwise

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return self.sub_expr.evaluate(df, context=context, mapping=mapping)

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return self.sub_expr.collect_agg(depth=depth)


def extract_partition_counts(
    exprs: Sequence[Expr],
    child_ir_count: int,
    *,
    update: MutableMapping[Expr, int] | None = None,
    skip_fused_exprs: bool = False,
) -> MutableMapping[Expr, int]:
    """
    Extract a partition-count mapping for Expr nodes.

    Parameters
    ----------
    exprs
        Sequence of root expressions to traverse and
        get partition counts.
    child_ir_count
        Partition count for the child-IR node.
    update
        Existing mapping to update.
    skip_fused_exprs
        Whether to skip over FusedExpr objects. This
        can be used to stay within a local FusedExpr
        sub-expression.

    Returns
    -------
    Mapping between Expr nodes and partition counts.
    """
    expr_partition_counts: MutableMapping[Expr, int] = update or {}
    for expr in exprs:
        for node in list(traversal([expr]))[::-1]:
            if isinstance(node, FusedExpr):
                # Process the fused sub-expression graph first
                if skip_fused_exprs:
                    continue  # Stay within the current sub expression
                expr_partition_counts = extract_partition_counts(
                    [node.sub_expr],
                    child_ir_count,
                    update=expr_partition_counts,
                    skip_fused_exprs=True,
                )
                expr_partition_counts[node] = expr_partition_counts[node.sub_expr]
            elif isinstance(node, Agg):
                # Assume all aggregations produce 1 partition
                expr_partition_counts[node] = 1
            elif node.is_pointwise:
                # Pointwise expressions should preserve child partition count
                if node.children:
                    # Assume maximum child partition count
                    expr_partition_counts[node] = max(
                        [expr_partition_counts[c] for c in node.children]
                    )
                else:
                    # If no children, we are preserving the child-IR partition count
                    expr_partition_counts[node] = child_ir_count
            else:
                raise NotImplementedError(
                    f"{type(node)} not supported for multiple partitions."
                )

    return expr_partition_counts


def replace_sub_expr(e: Expr, rec: ExprTransformer):
    """Replace a target expression node."""
    mapping = rec.state["mapping"]
    if e in mapping:
        return mapping[e]
    return reuse_if_unchanged(e, rec)


def rename_agg(agg: Agg, new_name: str):
    """Modify the name of an aggregation expression."""
    return CachingVisitor(
        replace_sub_expr,
        state={"mapping": {agg: Agg(agg.dtype, new_name, agg.options, *agg.children)}},
    )(agg)


def decompose_expr_graph(expr: Expr) -> FusedExpr:
    """Transform an Expr into a graph of FusedExpr nodes."""
    root = expr
    while True:
        exprs = [
            e
            for e in list(traversal([root]))[::-1]
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


def evaluate_chunk(
    df: DataFrame,
    expr: Expr,
    children: tuple[Expr, ...],
    *references: Column,
) -> Column:
    """Evaluate a single aggregation."""
    return expr.evaluate(df, mapping=dict(zip(children, references, strict=False)))


def evaluate_chunk_multi(
    df: DataFrame,
    exprs: Sequence[Expr],
    children: tuple[Expr, ...],
    *references: Column,
) -> tuple[Column, ...]:
    """Evaluate multiple aggregations."""
    return tuple(
        expr.evaluate(df, mapping=dict(zip(children, references, strict=False)))
        for expr in exprs
    )


def combine_chunks_multi(
    column_chunks: Sequence[tuple[Column]],
    combine_aggs: Sequence[Agg],
    finalize: tuple[plc.DataType, str] | None,
) -> Column:
    """Aggregate Column chunks."""
    column_chunk_lists = zip(*column_chunks, strict=False)

    combined = [
        agg.op(
            Column(
                plc.concatenate.concatenate([col.obj for col in column_chunk_list]),
                name=column_chunk_list[0].name,
            )
        )
        for agg, column_chunk_list in zip(
            combine_aggs, column_chunk_lists, strict=False
        )
    ]

    if finalize:
        # Perform optional BinOp on combined columns
        dt, op_name = finalize
        op = getattr(plc.binaryop.BinaryOperator, op_name)
        cols = [c.obj for c in combined]
        return Column(plc.binaryop.binary_operation(*cols, op, dt))

    assert len(combined) == 1
    return combined[0]


def make_agg_graph(
    expr: FusedExpr,
    child_name: str,
    expr_partition_counts: MutableMapping[Expr, int],
) -> MutableMapping[Any, Any]:
    """Build a FusedExpr aggregation graph."""
    agg = expr.sub_expr
    assert isinstance(agg, Agg), f"Expected Agg, got {agg}"
    assert agg.name in _SUPPORTED_AGGS, f"Agg {agg} not supported"

    # NOTE: This algorithm assumes we are doing nested
    # aggregations, or we are only aggregating a single
    # column. If we are performing aligned aggregations
    # across multiple columns at once, we should perform
    # our reduction at the DataFrame level instead.

    key_name = get_key_name(expr)
    expr_child_names = [get_key_name(c) for c in expr.children]
    expr_bcast = [expr_partition_counts[c] == 1 for c in expr.children]
    input_count = max(expr_partition_counts[c] for c in agg.children)

    graph: MutableMapping[Any, Any] = {}

    # Define operations for each aggregation stage
    agg_name = agg.name
    if agg_name == "count":
        chunk_aggs = [agg]
        combine_aggs = [rename_agg(agg, "sum")]
        finalize = None
    elif agg_name == "mean":
        sum_agg = rename_agg(agg, "sum")
        count_agg = rename_agg(agg, "count")
        chunk_aggs = [sum_agg, count_agg]
        combine_aggs = [sum_agg, sum_agg]
        finalize = (agg.dtype, "DIV")
    else:
        chunk_aggs = [agg]
        combine_aggs = [agg]
        finalize = None

    # Pointwise stage
    chunk_name = f"chunk-{key_name}"
    for i in range(input_count):
        graph[(chunk_name, i)] = (
            evaluate_chunk_multi,
            (child_name, i),
            chunk_aggs,
            expr.children,
            *[
                (name, 0) if bcast else (name, i)
                for name, bcast in zip(expr_child_names, expr_bcast, strict=False)
            ],
        )

    # Combine and finalize
    graph[(key_name, 0)] = (
        combine_chunks_multi,
        list(graph.keys()),
        combine_aggs,
        finalize,
    )

    return graph


def make_pointwise_graph(
    expr: FusedExpr,
    child_name: str,
    expr_partition_counts: MutableMapping[Expr, int],
) -> MutableMapping[Any, Any]:
    """Build simple pointwise FusedExpr graph."""
    key_name = get_key_name(expr)
    expr_child_names = [get_key_name(c) for c in expr.children]
    expr_bcast = [expr_partition_counts[c] == 1 for c in expr.children]
    count = expr_partition_counts[expr]
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


def make_fusedexpr_graph(
    expr: FusedExpr,
    child_name: str,
    expr_partition_counts: MutableMapping[Expr, int],
) -> MutableMapping[Any, Any]:
    """Build task graph for a FusedExpr node."""
    sub_expr = expr.sub_expr
    if isinstance(sub_expr, Agg) and sub_expr.name in _SUPPORTED_AGGS:
        return make_agg_graph(expr, child_name, expr_partition_counts)
    elif expr.is_pointwise:
        return make_pointwise_graph(expr, child_name, expr_partition_counts)
    else:
        # TODO: Implement "complex" aggs (e.g. mean, std, etc)
        raise NotImplementedError(
            f"{type(sub_expr)} not supported for multiple partitions."
        )
