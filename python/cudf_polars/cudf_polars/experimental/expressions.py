# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition Expr classes and utilities."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.dsl.expressions.aggregation import Agg
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.binaryop import BinOp
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.dsl.expressions.unary import Cast
from cudf_polars.dsl.traversal import (
    CachingVisitor,
    traversal,
)
from cudf_polars.experimental.base import get_key_name
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from cudf_polars.dsl.expressions.base import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.typing import ExprTransformer


_SUPPORTED_AGGS = ("count", "min", "max", "sum", "mean", "n_unique")


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
    """

    __slots__ = ("kind", "sub_expr")
    _non_child = ("dtype", "sub_expr", "kind")

    def __init__(
        self,
        dtype: plc.DataType,
        sub_expr: Expr,
        kind: str | None,
        *children: FusedExpr,
    ):
        self.dtype = dtype
        self.sub_expr = sub_expr
        self.kind = kind
        self.children = children
        self.is_pointwise = self.kind == "pointwise"
        assert all(isinstance(c, FusedExpr) for c in children)
        assert kind in ("pointwise", "shuffle", "aggregation")

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return self.sub_expr.evaluate(df, context=context, mapping=mapping)


class NoOp(Expr):
    """No-op expression."""

    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(self, dtype: plc.DataType, value: Expr) -> None:
        self.dtype = dtype
        self.children = (value,)
        self.is_pointwise = True

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        return child.evaluate(df, context=context, mapping=mapping)


class ShuffleColumn(Expr):
    """Shuffle expression."""

    __slots__ = ("config_options",)
    _non_child = ("dtype", "config_options")

    def __init__(
        self,
        dtype: plc.DataType,
        config_options: ConfigOptions,
        value: Expr,
    ) -> None:
        self.dtype = dtype
        self.config_options = config_options
        self.children = (value,)
        self.is_pointwise = False


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
    cutoff_types = (FusedExpr,) if skip_fused_exprs else ()
    for expr in exprs:
        for node in list(traversal([expr], cutoff_types=cutoff_types))[::-1]:
            if isinstance(node, FusedExpr):
                # Process the fused sub-expression graph first
                expr_partition_counts = extract_partition_counts(
                    [node.sub_expr],
                    child_ir_count,
                    update=expr_partition_counts,
                    skip_fused_exprs=True,
                )
                expr_partition_counts[node] = expr_partition_counts[node.sub_expr]
            elif isinstance(node, (Agg, Literal)):
                # Assume all aggregations produce 1 partition
                expr_partition_counts[node] = 1
            elif node.is_pointwise or isinstance(node, ShuffleColumn):
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


def _decompose(expr: Expr, rec: ExprTransformer) -> FusedExpr:
    # Used by `decompose_expr_graph`

    # Transform child expressions first
    new_children = tuple(map(rec, expr.children))
    fused_children: list[FusedExpr] = []
    if new_children:
        # Non-leaf node.
        # Construct child lists for new expressions
        # (both the fused expression and the sub-expression)
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
        sub_expr = expr.reconstruct(sub_expr_children)
    else:
        # Leaf node.
        # Convert to simple FusedExpr with no children
        sub_expr = expr

    return construct_fused_expr(sub_expr, fused_children)


def construct_fused_expr(sub_expr: Expr, fused_children: list[FusedExpr]) -> FusedExpr:
    """
    Construct new FusedExpr object.

    Parameters
    ----------
    sub_expr
        Expression to be wrapped in a ``FusedExpr`` class.
    fused_children
        Children of ``sub_expr`` that are already ``FusedExpr`` nodes.

    Returns
    -------
    New ``FusedExpr`` object.
    """
    if sub_expr.is_pointwise:
        # Pointwise expressions are always supported.
        kind = "pointwise"
        final_expr = sub_expr
    elif isinstance(sub_expr, Agg) and sub_expr.name in _SUPPORTED_AGGS:
        # This is a supported Agg expression.
        kind = "aggregation"
        agg = sub_expr
        agg_name = agg.name
        chunk_expr: Expr
        if agg_name == "count":
            # Chunkwise
            chunk_expr = agg
            # Combine
            combine_expr = Agg(
                agg.dtype,
                "sum",
                None,
                chunk_expr,
            )
            # Finalize
            final_expr = NoOp(agg.dtype, combine_expr)
        elif agg_name == "mean":
            # Chunkwise
            chunk_exprs = [
                Agg(agg.dtype, "sum", None, *agg.children),
                Agg(agg.dtype, "count", None, *agg.children),
            ]
            # Combine
            combine_exprs = [
                Agg(
                    agg.dtype,
                    "sum",
                    None,
                    chunk_expr,
                )
                for chunk_expr in chunk_exprs
            ]
            # Finalize
            final_expr = BinOp(
                agg.dtype,
                plc.binaryop.BinaryOperator.DIV,
                *combine_exprs,
            )
        elif agg_name == "n_unique":
            # Inject shuffle
            # TODO: Avoid shuffle if possible
            (child,) = agg.children
            shuffled = FusedExpr(
                child.dtype,
                ShuffleColumn(child.dtype, ConfigOptions({}), child),
                "shuffle",
                *fused_children,
            )
            fused_children = [shuffled]
            # Chunkwise
            chunk_expr = Cast(agg.dtype, shuffled)
            # Combine
            combine_expr = Agg(agg.dtype, "sum", None, chunk_expr)
            # Finalize
            final_expr = NoOp(agg.dtype, combine_expr)
        else:
            # Chunkwise
            chunk_expr = agg
            # Combine
            combine_expr = Agg(
                agg.dtype,
                agg.name,
                agg.options,
                chunk_expr,
            )
            # Finalize
            final_expr = NoOp(agg.dtype, combine_expr)
    else:
        # This is an un-supported expression - raise.
        raise NotImplementedError(
            f"{type(sub_expr)} not supported for multiple partitions."
        )

    return FusedExpr(final_expr.dtype, final_expr, kind, *fused_children)


def decompose_expr_graph(expr: Expr) -> Expr:
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
    mapping = dict(zip(children, references, strict=True))
    return tuple(expr.evaluate(df, mapping=mapping) for expr in exprs)


def combine_chunks_multi_agg(
    column_chunks: Sequence[tuple[Column, ...]],
    combine_aggs: Sequence[Agg],
    finalize: Expr,
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

    if isinstance(finalize, NoOp):
        (col,) = combined
    else:
        col = finalize.evaluate(
            DataFrame([]),  # Placeholder DataFrame
            mapping=dict(zip(finalize.children, combined, strict=True)),
        )

    return col.rename(name)


def make_agg_graph(
    named_expr: NamedExpr,
    expr_partition_counts: MutableMapping[Expr, int],
    child_ir: IR,
) -> MutableMapping[Any, Any]:
    """Build a FusedExpr aggregation graph."""
    expr = named_expr.value
    assert isinstance(expr, FusedExpr)
    assert expr.kind == "aggregation"

    # Define aggregation steps
    final_expr = expr.sub_expr
    combine_exprs = final_expr.children
    chunkwise_exprs = tuple(
        chain.from_iterable(combine_expr.children for combine_expr in combine_exprs)
    )

    # NOTE: This algorithm assumes we are doing nested
    # aggregations, or we are only aggregating a single
    # column. If we are performing aligned aggregations
    # across multiple columns at once, we should perform
    # our reduction at the DataFrame level instead.

    key_name = get_key_name(expr)
    expr_child_names = [get_key_name(c) for c in expr.children]
    expr_bcast = [expr_partition_counts[c] == 1 for c in expr.children]
    input_count = max(expr_partition_counts[c] for c in chunkwise_exprs[0].children)

    graph: MutableMapping[Any, Any] = {}

    # Pointwise stage
    pointwise_keys = []
    key_name = get_key_name(expr)
    child_name = get_key_name(child_ir)
    chunk_name = f"chunk-{key_name}"
    for i in range(input_count):
        pointwise_keys.append((chunk_name, i))
        graph[pointwise_keys[-1]] = (
            evaluate_chunk_multi_agg,
            (child_name, i),
            chunkwise_exprs,
            expr.children,
            *(
                (name, 0) if bcast else (name, i)
                for name, bcast in zip(expr_child_names, expr_bcast, strict=True)
            ),
        )

    # Combine and finalize
    graph[(key_name, 0)] = (
        combine_chunks_multi_agg,
        pointwise_keys,
        combine_exprs,
        final_expr,
        named_expr.name,
    )

    return graph


def make_shuffle_graph(
    named_expr: NamedExpr,
    expr_partition_counts: MutableMapping[Expr, int],
    child_ir: IR,
    child_ir_partition_info: PartitionInfo,
) -> MutableMapping[Any, Any]:
    """Build a FusedExpr aggregation graph for shuffling."""
    # TODO: Add shuffle graph logic
    raise NotImplementedError()


def make_pointwise_graph(
    named_expr: NamedExpr,
    expr_partition_counts: MutableMapping[Expr, int],
    child: IR,
) -> MutableMapping[Any, Any]:
    """Build simple pointwise FusedExpr graph."""
    expr = named_expr.value
    child_name = get_key_name(child)
    key_name = get_key_name(expr)
    expr_child_names = [get_key_name(c) for c in expr.children]
    expr_bcast = [expr_partition_counts[c] == 1 for c in expr.children]
    count = expr_partition_counts[expr]
    assert isinstance(expr, FusedExpr)
    sub_expr = named_expr.reconstruct(expr.sub_expr)
    return {
        (key_name, i): (
            evaluate_chunk,
            (child_name, i),
            sub_expr,
            expr.children,
            *(
                (name, 0) if bcast else (name, i)
                for name, bcast in zip(expr_child_names, expr_bcast, strict=True)
            ),
        )
        for i in range(count)
    }


def make_fusedexpr_graph(
    named_expr: NamedExpr,
    expr_partition_counts: MutableMapping[Expr, int],
    child: IR,
    child_partition_info: PartitionInfo,
) -> MutableMapping[Any, Any]:
    """Build task graph for a FusedExpr node."""
    expr = named_expr.value
    assert isinstance(expr, FusedExpr)
    if expr.kind == "pointwise":
        return make_pointwise_graph(named_expr, expr_partition_counts, child)
    elif expr.kind == "shuffle":
        return make_shuffle_graph(
            named_expr, expr_partition_counts, child, child_partition_info
        )
    elif expr.kind == "aggregation":
        return make_agg_graph(named_expr, expr_partition_counts, child)
    else:
        raise ValueError()
