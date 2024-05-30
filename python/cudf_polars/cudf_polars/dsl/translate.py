# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Translate polars IR representation to ours."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from functools import singledispatch
from typing import Any

from polars.polars import _expr_nodes as pl_expr, _ir_nodes as pl_ir

import cudf._lib.pylibcudf as plc  # noqa: TCH002, singledispatch register needs this name defined.

from cudf_polars.dsl import expr, ir
from cudf_polars.utils import dtypes

__all__ = ["translate_ir", "translate_expr"]


class set_node(AbstractContextManager):
    """Run a block with current node set in the visitor."""

    __slots__ = ("n", "visitor")

    def __init__(self, visitor, n: int):
        self.visitor = visitor
        self.n = n

    def __enter__(self):
        n = self.visitor.get_node()
        self.visitor.set_node(self.n)
        self.n = n

    def __exit__(self, *args):
        self.visitor.set_node(self.n)


noop_context: nullcontext = nullcontext()


@singledispatch
def _translate_ir(node: Any, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    raise NotImplementedError(f"Translation for {type(node).__name__}")


@_translate_ir.register
def _(node: pl_ir.PythonScan, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    return ir.PythonScan(
        schema,
        node.options,
        translate_expr(visitor, n=node.predicate)
        if node.predicate is not None
        else None,
    )


@_translate_ir.register
def _(node: pl_ir.Scan, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    return ir.Scan(
        schema,
        node.scan_type,
        node.paths,
        node.file_options,
        translate_expr(visitor, n=node.predicate)
        if node.predicate is not None
        else None,
    )


@_translate_ir.register
def _(node: pl_ir.Cache, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    return ir.Cache(schema, node.id_, translate_ir(visitor, n=node.input))


@_translate_ir.register
def _(
    node: pl_ir.DataFrameScan, visitor: Any, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.DataFrameScan(
        schema,
        node.df,
        node.projection,
        translate_expr(visitor, n=node.selection)
        if node.selection is not None
        else None,
    )


@_translate_ir.register
def _(node: pl_ir.Select, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
    cse_exprs = [translate_expr(visitor, n=e) for e in node.cse_expr]
    exprs = [translate_expr(visitor, n=e) for e in node.expr]
    return ir.Select(schema, inp, cse_exprs, exprs)


@_translate_ir.register
def _(node: pl_ir.GroupBy, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
    aggs = [translate_expr(visitor, n=e) for e in node.aggs]
    keys = [translate_expr(visitor, n=e) for e in node.keys]
    return ir.GroupBy(
        schema,
        inp,
        aggs,
        keys,
        node.maintain_order,
        node.options,
    )


@_translate_ir.register
def _(node: pl_ir.Join, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    # Join key dtypes are dependent on the schema of the left and
    # right inputs, so these must be translated with the relevant
    # input active.
    with set_node(visitor, node.input_left):
        inp_left = translate_ir(visitor, n=None)
        left_on = [translate_expr(visitor, n=e) for e in node.left_on]
    with set_node(visitor, node.input_right):
        inp_right = translate_ir(visitor, n=None)
        right_on = [translate_expr(visitor, n=e) for e in node.right_on]
    return ir.Join(schema, inp_left, inp_right, left_on, right_on, node.options)


@_translate_ir.register
def _(node: pl_ir.HStack, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
    cse_exprs = [translate_expr(visitor, n=e) for e in node.cse_exprs]
    exprs = [translate_expr(visitor, n=e) for e in node.exprs]
    return ir.HStack(schema, inp, cse_exprs, exprs)


@_translate_ir.register
def _(node: pl_ir.Reduce, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
    exprs = [translate_expr(visitor, n=e) for e in node.expr]
    return ir.Reduce(schema, inp, exprs)


@_translate_ir.register
def _(node: pl_ir.Distinct, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    return ir.Distinct(
        schema,
        translate_ir(visitor, n=node.input),
        node.options,
    )


@_translate_ir.register
def _(node: pl_ir.Sort, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
    by = [translate_expr(visitor, n=e) for e in node.by_column]
    return ir.Sort(schema, inp, by, node.sort_options, node.slice)


@_translate_ir.register
def _(node: pl_ir.Slice, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    return ir.Slice(schema, translate_ir(visitor, n=node.input), node.offset, node.len)


@_translate_ir.register
def _(node: pl_ir.Filter, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
    mask = translate_expr(visitor, n=node.predicate)
    return ir.Filter(schema, inp, mask)


@_translate_ir.register
def _(
    node: pl_ir.SimpleProjection, visitor: Any, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.Projection(schema, translate_ir(visitor, n=node.input))


@_translate_ir.register
def _(node: pl_ir.MapFunction, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    name, *options = node.function
    return ir.MapFunction(
        schema,
        # TODO: merge_sorted breaks this pattern
        translate_ir(visitor, n=node.input),
        name,
        options,
    )


@_translate_ir.register
def _(node: pl_ir.Union, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    return ir.Union(
        schema, [translate_ir(visitor, n=n) for n in node.inputs], node.options
    )


@_translate_ir.register
def _(node: pl_ir.HConcat, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    return ir.HConcat(schema, [translate_ir(visitor, n=n) for n in node.inputs])


@_translate_ir.register
def _(node: pl_ir.ExtContext, visitor: Any, schema: dict[str, plc.DataType]) -> ir.IR:
    return ir.ExtContext(
        schema,
        translate_ir(visitor, n=node.input),
        [translate_ir(visitor, n=n) for n in node.contexts],
    )


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
        schema = {k: dtypes.from_polars(v) for k, v in visitor.get_schema().items()}
        return _translate_ir(node, visitor, schema)


@singledispatch
def _translate_expr(node: Any, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    raise NotImplementedError(f"Translation for {type(node).__name__}")


@_translate_expr.register
def _(node: pl_expr.PyExprIR, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    e = translate_expr(visitor, n=node.node)
    return expr.NamedExpr(dtype, node.output_name, e)


@_translate_expr.register
def _(node: pl_expr.Function, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    name, *options = node.function_data
    options = tuple(options)
    if isinstance(name, pl_expr.StringFunction):
        return expr.StringFunction(
            dtype,
            name,
            options,
            *(translate_expr(visitor, n=n) for n in node.input),
        )
    elif isinstance(name, pl_expr.BooleanFunction):
        return expr.BooleanFunction(
            dtype,
            name,
            options,
            *(translate_expr(visitor, n=n) for n in node.input),
        )
    else:
        raise NotImplementedError(f"No handler for Expr function node with {name=}")


@_translate_expr.register
def _(node: pl_expr.Window, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    # TODO: raise in groupby?
    if node.partition_by is None:
        return expr.RollingWindow(
            dtype, node.options, translate_expr(visitor, n=node.function)
        )
    else:
        return expr.GroupedRollingWindow(
            dtype,
            node.options,
            translate_expr(visitor, n=node.function),
            *(translate_expr(visitor, n=n) for n in node.partition_by),
        )


@_translate_expr.register
def _(node: pl_expr.Literal, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    return expr.Literal(dtype, node.value)


@_translate_expr.register
def _(node: pl_expr.Sort, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    # TODO: raise in groupby
    return expr.Sort(dtype, node.options, translate_expr(visitor, n=node.expr))


@_translate_expr.register
def _(node: pl_expr.SortBy, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    return expr.SortBy(
        dtype,
        node.sort_options,
        translate_expr(visitor, n=node.expr),
        *(translate_expr(visitor, n=n) for n in node.by),
    )


@_translate_expr.register
def _(node: pl_expr.Gather, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    return expr.Gather(
        dtype,
        translate_expr(visitor, n=node.expr),
        translate_expr(visitor, n=node.idx),
    )


@_translate_expr.register
def _(node: pl_expr.Filter, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    return expr.Filter(
        dtype,
        translate_expr(visitor, n=node.input),
        translate_expr(visitor, n=node.by),
    )


@_translate_expr.register
def _(node: pl_expr.Cast, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    inner = translate_expr(visitor, n=node.expr)
    # Push casts into literals so we can handle Cast(Literal(Null))
    if isinstance(inner, expr.Literal):
        return expr.Literal(dtype, inner.value)
    else:
        return expr.Cast(dtype, inner)


@_translate_expr.register
def _(node: pl_expr.Column, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    return expr.Col(dtype, node.name)


@_translate_expr.register
def _(node: pl_expr.Agg, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    return expr.Agg(
        dtype,
        node.name,
        node.options,
        translate_expr(visitor, n=node.arguments),
    )


@_translate_expr.register
def _(node: pl_expr.BinaryExpr, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    return expr.BinOp(
        dtype,
        expr.BinOp._MAPPING[node.op],
        translate_expr(visitor, n=node.left),
        translate_expr(visitor, n=node.right),
    )


@_translate_expr.register
def _(node: pl_expr.Len, visitor: Any, dtype: plc.DataType) -> expr.Expr:
    return expr.Len(dtype)


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
        # TODO: type narrowing doesn't rule out int since PyExprIR is Unknown
        assert not isinstance(n, int)
        node = n
        dtype = dtypes.from_polars(visitor.get_dtype(node.node))
    else:
        node = visitor.view_expression(n)
        dtype = dtypes.from_polars(visitor.get_dtype(n))
    return _translate_expr(node, visitor, dtype)
