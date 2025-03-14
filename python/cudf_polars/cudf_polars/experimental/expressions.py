# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition Expr classes and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.aggregation import Agg
from cudf_polars.dsl.expressions.base import Expr, NamedExpr
from cudf_polars.dsl.traversal import (
    CachingVisitor,
    reuse_if_unchanged,
    traversal,
)
from cudf_polars.experimental.base import get_key_name

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.typing import ExprTransformer


_SUPPORTED_AGGS = ("count", "min", "max", "sum", "mean")


class FusedExpr(Expr):
    """
    A single fused component of a decomposed Expr graph.

    Notes
    -----
    - A FusedExpr object points to a single node in a
    decomposed Expr graph (i.e. ``sub_expr``).
    - A FusedExpr object may have children, but those
    children must be other FusedExpr objects.
    - When a FusedExpr object is evaluated, it will
    substitute it's evaluated children into ``sub_expr``,
    and evaluate the re-written sub-expression.
    - A FusedExpr object may point to a non-pointwise
    Expr node. However, all other nodes in ``sub_expr``
    must be pointwise or FusedExpr children.
    """

    __slots__ = ("sub_expr",)
    _non_child = ("dtype", "sub_expr")

    def __init__(
        self,
        dtype: plc.DataType,
        sub_expr: Expr,
        *children: FusedExpr,
    ):
        self.dtype = dtype
        self.sub_expr = sub_expr
        assert all(isinstance(c, FusedExpr) for c in children)
        self.children = children
        self.is_pointwise = sub_expr.is_pointwise
        assert all(
            e.is_pointwise or isinstance(e, FusedExpr)
            for e in traversal(list(sub_expr.children))
        ), f"Invalid FusedExpr sub-expression: {sub_expr}"


def extract_partition_counts(
    exprs: Sequence[Expr | NamedExpr],
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
    for e in exprs:
        expr = e.value if isinstance(e, NamedExpr) else e
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
            else:  # pragma: no cover
                raise NotImplementedError(
                    f"{type(node)} not supported for multiple partitions."
                )

    return expr_partition_counts


def _replace(e: Expr, rec: ExprTransformer) -> Expr:
    return rec.state["mapping"].get(e, reuse_if_unchanged(e, rec))


def replace(e: Expr, mapping: Mapping[Expr, Expr]) -> Expr:
    """Replace one or more expression nodes."""
    mapper = CachingVisitor(_replace, state={"mapping": mapping})
    return mapper(e)


def rename_agg(agg: Agg, new_name: str, *, new_options: Any = None):
    """Modify the name of an aggregation expression."""
    return replace(agg, {agg: Agg(agg.dtype, new_name, new_options, *agg.children)})


def _decompose(expr: Expr, rec: ExprTransformer) -> FusedExpr:
    # Used by `decompose_expr_graph`

    # Transform child expressions first
    new_children = tuple(map(rec, expr.children))

    if new_children:
        # Non-leaf node.
        # Construct child lists for new expressions
        # (both the fused expression and the sub-expression)
        fused_children: list[FusedExpr] = []
        sub_expr_children: list[Expr] = []
        for child in new_children:
            # All children should be FusedExpr
            assert isinstance(child, FusedExpr), "FusedExpr children must be FusedExpr."
            if child.is_pointwise:
                # Pointwise children must be fused into the
                # "new" FusedExpr node with root `expr`
                for c in child.children:
                    assert isinstance(c, FusedExpr), (
                        "FusedExpr children must be FusedExpr."
                    )
                    fused_children.append(c)
                sub_expr_children.append(child.sub_expr)
            else:
                # Non-pointwise children must remain as
                # distinct FusedExpr nodes
                fused_children.append(child)
                sub_expr_children.append(child)
        # Reconstruct and return the new FusedExpr
        sub_expr = expr.reconstruct(sub_expr_children)
        return FusedExpr(sub_expr.dtype, sub_expr, *fused_children)
    else:
        # Leaf node.
        # Convert to simple FusedExpr with no children
        return FusedExpr(expr.dtype, expr)


def decompose_expr_graph(expr):
    """Transform an Expr into a graph of FusedExpr nodes."""
    mapper = CachingVisitor(_decompose)
    return mapper(expr)


def evaluate_chunk(
    df: DataFrame,
    expr: Expr,
    children: tuple[Expr, ...],
    *references: Column,
) -> Column:
    """Evaluate the sub-expression of a simple FusedExpr node."""
    return expr.evaluate(df, mapping=dict(zip(children, references, strict=True)))


def evaluate_chunk_multi_agg(
    df: DataFrame,
    exprs: Sequence[Expr],
    children: tuple[Expr, ...],
    *references: Column,
) -> tuple[Column, ...]:
    """Evaluate multiple aggregations."""
    return tuple(
        expr.evaluate(df, mapping=dict(zip(children, references, strict=True)))
        for expr in exprs
    )


def combine_chunks_multi_agg(
    column_chunks: Sequence[tuple[Column]],
    combine_aggs: Sequence[Agg],
    finalize: tuple[plc.DataType, str] | None,
    name: str,
) -> Column:
    """Aggregate Column chunks."""
    column_chunk_lists = zip(*column_chunks, strict=True)

    combined = [
        agg.op(
            Column(
                plc.concatenate.concatenate([col.obj for col in column_chunk_list]),
                name=column_chunk_list[0].name,
            )
        )
        for agg, column_chunk_list in zip(combine_aggs, column_chunk_lists, strict=True)
    ]

    if finalize:
        # Perform optional BinOp on combined columns
        dt, op_name = finalize
        op = getattr(plc.binaryop.BinaryOperator, op_name)
        col = Column(plc.binaryop.binary_operation(*(c.obj for c in combined), op, dt))
    else:
        assert len(combined) == 1
        col = combined[0]

    return col.rename(name)


def make_agg_graph(
    named_expr: NamedExpr,
    child_name: str,
    expr_partition_counts: MutableMapping[Expr, int],
) -> MutableMapping[Any, Any]:
    """Build a FusedExpr aggregation graph."""
    expr = named_expr.value
    assert isinstance(expr, FusedExpr)
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
            evaluate_chunk_multi_agg,
            (child_name, i),
            chunk_aggs,
            expr.children,
            *[
                (name, 0) if bcast else (name, i)
                for name, bcast in zip(expr_child_names, expr_bcast, strict=True)
            ],
        )

    # Combine and finalize
    graph[(key_name, 0)] = (
        combine_chunks_multi_agg,
        list(graph.keys()),
        combine_aggs,
        finalize,
        named_expr.name,
    )

    return graph


def make_pointwise_graph(
    named_expr: NamedExpr,
    child_name: str,
    expr_partition_counts: MutableMapping[Expr, int],
) -> MutableMapping[Any, Any]:
    """Build simple pointwise FusedExpr graph."""
    expr = named_expr.value
    assert isinstance(expr, FusedExpr)
    key_name = get_key_name(expr)
    expr_child_names = [get_key_name(c) for c in expr.children]
    expr_bcast = [expr_partition_counts[c] == 1 for c in expr.children]
    count = expr_partition_counts[expr]
    sub_expr = named_expr.reconstruct(expr.sub_expr)
    return {
        (key_name, i): (
            evaluate_chunk,
            (child_name, i),
            sub_expr,
            expr.children,
            *[
                (name, 0) if bcast else (name, i)
                for name, bcast in zip(expr_child_names, expr_bcast, strict=True)
            ],
        )
        for i in range(count)
    }


def make_fusedexpr_graph(
    named_expr: NamedExpr,
    child_name: str,
    expr_partition_counts: MutableMapping[Expr, int],
) -> MutableMapping[Any, Any]:
    """Build task graph for a FusedExpr node."""
    expr = named_expr.value
    assert isinstance(expr, FusedExpr)
    sub_expr = expr.sub_expr
    if isinstance(sub_expr, Agg) and sub_expr.name in _SUPPORTED_AGGS:
        return make_agg_graph(named_expr, child_name, expr_partition_counts)
    elif expr.is_pointwise:
        return make_pointwise_graph(named_expr, child_name, expr_partition_counts)
    else:
        # TODO: Implement "complex" aggs (e.g. mean, std, etc)
        raise NotImplementedError(
            f"{type(sub_expr)} not supported for multiple partitions."
        )
