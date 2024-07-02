# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Translate polars IR representation to ours."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from functools import singledispatch
from typing import Any

import pyarrow as pa
from typing_extensions import assert_never

import polars.polars as plrs
from polars.polars import _expr_nodes as pl_expr, _ir_nodes as pl_ir

import cudf._lib.pylibcudf as plc

from cudf_polars.dsl import expr, ir
from cudf_polars.typing import NodeTraverser
from cudf_polars.utils import dtypes

__all__ = ["translate_ir", "translate_named_expr"]


class set_node(AbstractContextManager[None]):
    """
    Run a block with current node set in the visitor.

    Parameters
    ----------
    visitor
        The internal Rust visitor object
    n
        The node to set as the current root.

    Notes
    -----
    This is useful for translating expressions with a given node
    active, restoring the node when the block exits.
    """

    __slots__ = ("n", "visitor")
    visitor: NodeTraverser
    n: int

    def __init__(self, visitor: NodeTraverser, n: int) -> None:
        self.visitor = visitor
        self.n = n

    def __enter__(self) -> None:
        n = self.visitor.get_node()
        self.visitor.set_node(self.n)
        self.n = n

    def __exit__(self, *args: Any) -> None:
        self.visitor.set_node(self.n)


noop_context: nullcontext[None] = nullcontext()


@singledispatch
def _translate_ir(
    node: Any, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    raise NotImplementedError(
        f"Translation for {type(node).__name__}"
    )  # pragma: no cover


@_translate_ir.register
def _(
    node: pl_ir.PythonScan, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.PythonScan(
        schema,
        node.options,
        translate_named_expr(visitor, n=node.predicate)
        if node.predicate is not None
        else None,
    )


@_translate_ir.register
def _(
    node: pl_ir.Scan, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    typ, *options = node.scan_type
    return ir.Scan(
        schema,
        typ,
        tuple(options),
        node.paths,
        node.file_options,
        translate_named_expr(visitor, n=node.predicate)
        if node.predicate is not None
        else None,
    )


@_translate_ir.register
def _(
    node: pl_ir.Cache, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.Cache(schema, node.id_, translate_ir(visitor, n=node.input))


@_translate_ir.register
def _(
    node: pl_ir.DataFrameScan, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.DataFrameScan(
        schema,
        node.df,
        node.projection,
        translate_named_expr(visitor, n=node.selection)
        if node.selection is not None
        else None,
    )


@_translate_ir.register
def _(
    node: pl_ir.Select, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
        exprs = [translate_named_expr(visitor, n=e) for e in node.expr]
    return ir.Select(schema, inp, exprs, node.should_broadcast)


@_translate_ir.register
def _(
    node: pl_ir.GroupBy, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
        aggs = [translate_named_expr(visitor, n=e) for e in node.aggs]
        keys = [translate_named_expr(visitor, n=e) for e in node.keys]
    return ir.GroupBy(
        schema,
        inp,
        aggs,
        keys,
        node.maintain_order,
        node.options,
    )


@_translate_ir.register
def _(
    node: pl_ir.Join, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    # Join key dtypes are dependent on the schema of the left and
    # right inputs, so these must be translated with the relevant
    # input active.
    with set_node(visitor, node.input_left):
        inp_left = translate_ir(visitor, n=None)
        left_on = [translate_named_expr(visitor, n=e) for e in node.left_on]
    with set_node(visitor, node.input_right):
        inp_right = translate_ir(visitor, n=None)
        right_on = [translate_named_expr(visitor, n=e) for e in node.right_on]
    return ir.Join(schema, inp_left, inp_right, left_on, right_on, node.options)


@_translate_ir.register
def _(
    node: pl_ir.HStack, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
        exprs = [translate_named_expr(visitor, n=e) for e in node.exprs]
    return ir.HStack(schema, inp, exprs, node.should_broadcast)


@_translate_ir.register
def _(
    node: pl_ir.Reduce, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:  # pragma: no cover; polars doesn't emit this node yet
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
        exprs = [translate_named_expr(visitor, n=e) for e in node.expr]
    return ir.Reduce(schema, inp, exprs)


@_translate_ir.register
def _(
    node: pl_ir.Distinct, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.Distinct(
        schema,
        translate_ir(visitor, n=node.input),
        node.options,
    )


@_translate_ir.register
def _(
    node: pl_ir.Sort, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
        by = [translate_named_expr(visitor, n=e) for e in node.by_column]
    return ir.Sort(schema, inp, by, node.sort_options, node.slice)


@_translate_ir.register
def _(
    node: pl_ir.Slice, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.Slice(schema, translate_ir(visitor, n=node.input), node.offset, node.len)


@_translate_ir.register
def _(
    node: pl_ir.Filter, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    with set_node(visitor, node.input):
        inp = translate_ir(visitor, n=None)
        mask = translate_named_expr(visitor, n=node.predicate)
    return ir.Filter(schema, inp, mask)


@_translate_ir.register
def _(
    node: pl_ir.SimpleProjection,
    visitor: NodeTraverser,
    schema: dict[str, plc.DataType],
) -> ir.IR:
    return ir.Projection(schema, translate_ir(visitor, n=node.input))


@_translate_ir.register
def _(
    node: pl_ir.MapFunction, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    name, *options = node.function
    return ir.MapFunction(
        schema,
        # TODO: merge_sorted breaks this pattern
        translate_ir(visitor, n=node.input),
        name,
        options,
    )


@_translate_ir.register
def _(
    node: pl_ir.Union, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.Union(
        schema, [translate_ir(visitor, n=n) for n in node.inputs], node.options
    )


@_translate_ir.register
def _(
    node: pl_ir.HConcat, visitor: NodeTraverser, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.HConcat(schema, [translate_ir(visitor, n=n) for n in node.inputs])


def translate_ir(visitor: NodeTraverser, *, n: int | None = None) -> ir.IR:
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
    NotImplementedError
        If we can't translate the nodes due to unsupported functionality.
    """
    ctx: AbstractContextManager[None] = (
        set_node(visitor, n) if n is not None else noop_context
    )
    with ctx:
        node = visitor.view_current_node()
        schema = {k: dtypes.from_polars(v) for k, v in visitor.get_schema().items()}
        return _translate_ir(node, visitor, schema)


def translate_named_expr(
    visitor: NodeTraverser, *, n: pl_expr.PyExprIR
) -> expr.NamedExpr:
    """
    Translate a polars-internal named expression IR object into our representation.

    Parameters
    ----------
    visitor
        Polars NodeTraverser object
    n
        Node to translate, a named expression node.

    Returns
    -------
    Translated IR object.

    Notes
    -----
    The datatype of the internal expression will be obtained from the
    visitor by calling ``get_dtype``, for this to work properly, the
    caller should arrange that the expression is translated with the
    node that it references "active" for the visitor (see :class:`set_node`).

    Raises
    ------
    NotImplementedError
        If any translation fails due to unsupported functionality.
    """
    return expr.NamedExpr(n.output_name, translate_expr(visitor, n=n.node))


@singledispatch
def _translate_expr(
    node: Any, visitor: NodeTraverser, dtype: plc.DataType
) -> expr.Expr:
    raise NotImplementedError(
        f"Translation for {type(node).__name__}"
    )  # pragma: no cover


@_translate_expr.register
def _(node: pl_expr.Function, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
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
        if name == pl_expr.BooleanFunction.IsBetween:
            column, lo, hi = (translate_expr(visitor, n=n) for n in node.input)
            (closed,) = options
            lop, rop = expr.BooleanFunction._BETWEEN_OPS[closed]
            return expr.BinOp(
                dtype,
                plc.binaryop.BinaryOperator.LOGICAL_AND,
                expr.BinOp(dtype, lop, column, lo),
                expr.BinOp(dtype, rop, column, hi),
            )
        return expr.BooleanFunction(
            dtype,
            name,
            options,
            *(translate_expr(visitor, n=n) for n in node.input),
        )
    elif isinstance(name, pl_expr.TemporalFunction):
        return expr.TemporalFunction(
            dtype,
            name,
            options,
            *(translate_expr(visitor, n=n) for n in node.input),
        )
    elif isinstance(name, str):
        return expr.UnaryFunction(
            dtype,
            name,
            options,
            *(translate_expr(visitor, n=n) for n in node.input),
        )
    raise NotImplementedError(
        f"No handler for Expr function node with {name=}"
    )  # pragma: no cover; polars raises on the rust side for now


@_translate_expr.register
def _(node: pl_expr.Window, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    # TODO: raise in groupby?
    if isinstance(node.options, pl_expr.RollingGroupOptions):
        # pl.col("a").rolling(...)
        return expr.RollingWindow(
            dtype, node.options, translate_expr(visitor, n=node.function)
        )
    elif isinstance(node.options, pl_expr.WindowMapping):
        # pl.col("a").over(...)
        return expr.GroupedRollingWindow(
            dtype,
            node.options,
            translate_expr(visitor, n=node.function),
            *(translate_expr(visitor, n=n) for n in node.partition_by),
        )
    assert_never(node.options)


@_translate_expr.register
def _(node: pl_expr.Literal, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    if isinstance(node.value, plrs.PySeries):
        return expr.LiteralColumn(dtype, node.value)
    value = pa.scalar(node.value, type=plc.interop.to_arrow(dtype))
    return expr.Literal(dtype, value)


@_translate_expr.register
def _(node: pl_expr.Sort, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    # TODO: raise in groupby
    return expr.Sort(dtype, node.options, translate_expr(visitor, n=node.expr))


@_translate_expr.register
def _(node: pl_expr.SortBy, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    return expr.SortBy(
        dtype,
        node.sort_options,
        translate_expr(visitor, n=node.expr),
        *(translate_expr(visitor, n=n) for n in node.by),
    )


@_translate_expr.register
def _(node: pl_expr.Gather, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    return expr.Gather(
        dtype,
        translate_expr(visitor, n=node.expr),
        translate_expr(visitor, n=node.idx),
    )


@_translate_expr.register
def _(node: pl_expr.Filter, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    return expr.Filter(
        dtype,
        translate_expr(visitor, n=node.input),
        translate_expr(visitor, n=node.by),
    )


@_translate_expr.register
def _(node: pl_expr.Cast, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    inner = translate_expr(visitor, n=node.expr)
    # Push casts into literals so we can handle Cast(Literal(Null))
    if isinstance(inner, expr.Literal):
        return expr.Literal(dtype, inner.value.cast(plc.interop.to_arrow(dtype)))
    else:
        return expr.Cast(dtype, inner)


@_translate_expr.register
def _(node: pl_expr.Column, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    return expr.Col(dtype, node.name)


@_translate_expr.register
def _(node: pl_expr.Agg, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    return expr.Agg(
        dtype,
        node.name,
        node.options,
        *(translate_expr(visitor, n=n) for n in node.arguments),
    )


@_translate_expr.register
def _(node: pl_expr.Ternary, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    return expr.Ternary(
        dtype,
        translate_expr(visitor, n=node.predicate),
        translate_expr(visitor, n=node.truthy),
        translate_expr(visitor, n=node.falsy),
    )


@_translate_expr.register
def _(
    node: pl_expr.BinaryExpr, visitor: NodeTraverser, dtype: plc.DataType
) -> expr.Expr:
    return expr.BinOp(
        dtype,
        expr.BinOp._MAPPING[node.op],
        translate_expr(visitor, n=node.left),
        translate_expr(visitor, n=node.right),
    )


@_translate_expr.register
def _(node: pl_expr.Len, visitor: NodeTraverser, dtype: plc.DataType) -> expr.Expr:
    return expr.Len(dtype)


def translate_expr(visitor: NodeTraverser, *, n: int) -> expr.Expr:
    """
    Translate a polars-internal expression IR into our representation.

    Parameters
    ----------
    visitor
        Polars NodeTraverser object
    n
        Node to translate, an integer referencing a polars internal node.

    Returns
    -------
    Translated IR object.

    Raises
    ------
    NotImplementedError
        If any translation fails due to unsupported functionality.
    """
    node = visitor.view_expression(n)
    dtype = dtypes.from_polars(visitor.get_dtype(n))
    return _translate_expr(node, visitor, dtype)
