# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Multi-partition Expr classes and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.aggregation import Agg
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.traversal import toposort

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    import pylibcudf as plc

    from cudf_polars.containers import Column, DataFrame
    from cudf_polars.dsl.expressions.base import AggInfo


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


def get_expr_partition_count(
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
    expr_partition_count: MutableMapping[Expr, int] = update or {}
    for expr in exprs:
        for node in toposort(expr, reverse=True):
            if isinstance(node, FusedExpr):
                # Process the fused sub-expression graph first
                if skip_fused_exprs:
                    continue  # Stay within the current sub expression
                expr_partition_count = get_expr_partition_count(
                    [node.sub_expr],
                    child_ir_count,
                    update=expr_partition_count,
                    skip_fused_exprs=True,
                )
                expr_partition_count[node] = expr_partition_count[node.sub_expr]
            elif isinstance(node, Agg):
                # Assume all aggregations produce 1 partition
                expr_partition_count[node] = 1
            elif node.is_pointwise:
                # Pointwise expressions should preserve child partition count
                if node.children:
                    # Assume maximum child partition count
                    expr_partition_count[node] = max(
                        [expr_partition_count[c] for c in node.children]
                    )
                else:
                    # If no children, we are preserving the child-IR partition count
                    expr_partition_count[node] = child_ir_count
            else:
                raise NotImplementedError(
                    f"{type(node)} not supported for multiple partitions."
                )

    return expr_partition_count
