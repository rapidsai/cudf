# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Translate polars IR representation to ours."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from polars.polars import _expr_nodes as pl_expr, _ir_nodes as pl_ir

from cudf_polars.dsl import expr, ir

__all__ = ["translate_ir", "translate_expr"]


class set_node(AbstractContextManager):
    __slots__ = ("n", "visitor")

    def __init__(self, visitor, n):
        self.visitor = visitor
        self.n = n

    def __enter__(self):
        n = self.visitor.get_node()
        self.visitor.set_node(self.n)
        self.n = n

    def __exit__(self, *args):
        self.visitor.set_node(self.n)


noop_context: nullcontext = nullcontext()


def translate_ir(visitor: Any, *, n: int | None = None) -> ir.IR:
    """
    Translate a polars-internal IR node to our representation.

    Parameters
    ----------
    visitor
        Polars NodeTraverser object
    n
        Optional node to start traversing from, if not provided uses
        current polars-internal node.

    Returns
    -------
    Translated IR object

    Raises
    ------
    NotImplementedError if we can't translate the nodes due to
    unsupported functionality.
    """
    ctx: AbstractContextManager = (
        set_node(visitor, n) if n is not None else noop_context
    )
    with ctx:
        node = visitor.view_current_node()
        schema = visitor.get_schema()
        if isinstance(node, pl_ir.PythonScan):
            return ir.PythonScan(
                schema,
                node.options,
                translate_expr(visitor, n=node.predicate)
                if node.predicate is not None
                else None,
            )
        elif isinstance(node, pl_ir.Scan):
            return ir.Scan(
                schema,
                node.scan_type,
                node.paths,
                node.file_options,
                translate_expr(visitor, n=node.predicate)
                if node.predicate is not None
                else None,
            )
        elif isinstance(node, pl_ir.Cache):
            return ir.Cache(schema, node.id_, translate_ir(visitor, n=node.input))
        elif isinstance(node, pl_ir.DataFrameScan):
            return ir.DataFrameScan(
                schema,
                node.df,
                node.projection,
                translate_expr(visitor, n=node.selection)
                if node.selection is not None
                else None,
            )
        elif isinstance(node, pl_ir.Select):
            return ir.Select(
                schema,
                translate_ir(visitor, n=node.input),
                [translate_expr(visitor, n=e) for e in node.cse_expr],
                [translate_expr(visitor, n=e) for e in node.expr],
            )
        elif isinstance(node, pl_ir.GroupBy):
            return ir.GroupBy(
                schema,
                translate_ir(visitor, n=node.input),
                [translate_expr(visitor, n=e) for e in node.aggs],
                [translate_expr(visitor, n=e) for e in node.keys],
                node.options,
            )
        elif isinstance(node, pl_ir.Join):
            return ir.Join(
                schema,
                translate_ir(visitor, n=node.input_left),
                translate_ir(visitor, n=node.input_right),
                [translate_expr(visitor, n=e) for e in node.left_on],
                [translate_expr(visitor, n=e) for e in node.right_on],
                node.options,
            )
        elif isinstance(node, pl_ir.HStack):
            return ir.HStack(
                schema,
                translate_ir(visitor, n=node.input),
                [translate_expr(visitor, n=e) for e in node.exprs],
            )
        elif isinstance(node, pl_ir.Distinct):
            return ir.Distinct(
                schema,
                translate_ir(visitor, n=node.input),
                node.options,
            )
        elif isinstance(node, pl_ir.Sort):
            return ir.Sort(
                schema,
                translate_ir(visitor, n=node.input),
                [translate_expr(visitor, n=e) for e in node.by_column],
                node.sort_options,
            )
        elif isinstance(node, pl_ir.Slice):
            return ir.Slice(
                schema, translate_ir(visitor, n=node.input), node.offset, node.len
            )
        elif isinstance(node, pl_ir.Filter):
            return ir.Filter(
                schema,
                translate_ir(visitor, n=node.input),
                translate_expr(visitor, n=node.predicate),
            )
        elif isinstance(node, pl_ir.SimpleProjection):
            return ir.Projection(schema, translate_ir(visitor, n=node.input))
        elif isinstance(node, pl_ir.MapFunction):
            name, *options = node.function
            return ir.MapFunction(
                schema,
                # TODO: merge_sorted breaks this pattern
                translate_ir(visitor, n=node.input),
                name,
                options,
            )
        elif isinstance(node, pl_ir.Union):
            return ir.Union(schema, [translate_ir(visitor, n=n) for n in node.inputs])
        elif isinstance(node, pl_ir.HConcat):
            return ir.HConcat(schema, [translate_ir(visitor, n=n) for n in node.inputs])
        elif isinstance(node, pl_ir.ExtContext):
            return ir.ExtContext(
                schema,
                translate_ir(visitor, n=node.input),
                [translate_ir(visitor, n=n) for n in node.contexts],
            )
        else:
            raise NotImplementedError(
                f"No handler for LogicalPlan node with {type(node)=}"
            )


BOOLEAN_FUNCTIONS: frozenset[str] = frozenset()


def translate_expr(visitor: Any, *, n: int | pl_expr.PyExprIR) -> expr.Expr:
    """
    Translate a polars-internal expression IR into our representation.

    Parameters
    ----------
    visitor
        Polars NodeTraverser object
    n
        Node to translate, either an integer referencing a polars
        internal node, or a named expression node.

    Returns
    -------
    Translated IR object.

    Raises
    ------
    NotImplementedError if any translation fails due to unsupported functionality.
    """
    if isinstance(n, pl_expr.PyExprIR):
        # TODO: type narrowing didn't work because PyExprIR is Unknown
        assert not isinstance(n, int)
        return expr.NamedExpr(n.output_name, translate_expr(visitor, n=n.node))
    node = visitor.view_expression(n)
    if isinstance(node, pl_expr.Function):
        name, *options = node.function_data
        if name in BOOLEAN_FUNCTIONS:
            return expr.BooleanFunction(
                name,
                options,
                [translate_expr(visitor, n=n) for n in node.input],
            )
        else:
            raise NotImplementedError(f"No handler for Expr function node with {name=}")
    elif isinstance(node, pl_expr.Window):
        # TODO: raise in groupby?
        return expr.Window(
            translate_expr(visitor, n=node.function),
            [translate_expr(visitor, n=n) for n in node.partition_by]
            if node.partition_by is not None
            else None,
            node.options,
        )
    elif isinstance(node, pl_expr.Literal):
        return expr.Literal(node.dtype, node.value)
    elif isinstance(node, pl_expr.Sort):
        # TODO: raise in groupby
        return expr.Sort(translate_expr(visitor, n=node.expr), node.options)
    elif isinstance(node, pl_expr.SortBy):
        # TODO: raise in groupby
        return expr.SortBy(
            translate_expr(visitor, n=node.expr),
            [translate_expr(visitor, n=n) for n in node.by],
            node.descending,
        )
    elif isinstance(node, pl_expr.Gather):
        return expr.Gather(
            translate_expr(visitor, n=node.expr),
            translate_expr(visitor, n=node.idx),
        )
    elif isinstance(node, pl_expr.Filter):
        return expr.Filter(
            translate_expr(visitor, n=node.input),
            translate_expr(visitor, n=node.by),
        )
    elif isinstance(node, pl_expr.Cast):
        return expr.Cast(node.dtype, translate_expr(visitor, n=node.expr))
    elif isinstance(node, pl_expr.Column):
        return expr.Column(node.name)
    elif isinstance(node, pl_expr.Agg):
        return expr.Agg(
            translate_expr(visitor, n=node.arguments),
            node.name,
            node.options,
        )
    elif isinstance(node, pl_expr.BinaryExpr):
        return expr.BinOp(
            translate_expr(visitor, n=node.left),
            translate_expr(visitor, n=node.right),
            node.op,
        )
    elif isinstance(node, pl_expr.Len):
        return expr.Len()
    else:
        raise NotImplementedError(f"No handler for expression node with {type(node)=}")
