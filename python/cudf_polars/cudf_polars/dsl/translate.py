# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Translate polars IR representation to ours."""

from __future__ import annotations

import functools
import json
from contextlib import AbstractContextManager, nullcontext
from functools import singledispatch
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from typing_extensions import assert_never

import polars as pl
import polars.polars as plrs
from polars.polars import _expr_nodes as pl_expr, _ir_nodes as pl_ir

import pylibcudf as plc

from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.to_ast import insert_colrefs
from cudf_polars.typing import NodeTraverser
from cudf_polars.utils import dtypes, sorting

if TYPE_CHECKING:
    from polars import GPUEngine

    from cudf_polars.typing import NodeTraverser

__all__ = ["Translator", "translate_named_expr"]


class Translator:
    """
    Translates polars-internal IR nodes and expressions to our representation.

    Parameters
    ----------
    visitor
        Polars NodeTraverser object
    config
        GPU engine configuration.
    """

    def __init__(self, visitor: NodeTraverser, config: GPUEngine):
        self.visitor = visitor
        self.config = config
        self.errors: list[Exception] = []

    def translate_ir(self, *, n: int | None = None) -> ir.IR:
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
            If the version of Polars IR is unsupported.

        Notes
        -----
        Any expression nodes that cannot be translated are replaced by
        :class:`expr.ErrorNode` nodes and collected in the the `errors` attribute.
        After translation is complete, this list of errors should be inspected
        to determine if the query is supported.
        """
        ctx: AbstractContextManager[None] = (
            set_node(self.visitor, n) if n is not None else noop_context
        )
        # IR is versioned with major.minor, minor is bumped for backwards
        # compatible changes (e.g. adding new nodes), major is bumped for
        # incompatible changes (e.g. renaming nodes).
        if (version := self.visitor.version()) >= (4, 0):
            e = NotImplementedError(
                f"No support for polars IR {version=}"
            )  # pragma: no cover; no such version for now.
            self.errors.append(e)  # pragma: no cover
            raise e  # pragma: no cover

        with ctx:
            polars_schema = self.visitor.get_schema()
            try:
                schema = {k: dtypes.from_polars(v) for k, v in polars_schema.items()}
            except Exception as e:
                self.errors.append(NotImplementedError(str(e)))
                return ir.ErrorNode({}, str(e))
            try:
                node = self.visitor.view_current_node()
            except Exception as e:
                self.errors.append(e)
                return ir.ErrorNode(schema, str(e))
            try:
                result = _translate_ir(node, self, schema)
            except Exception as e:
                self.errors.append(e)
                return ir.ErrorNode(schema, str(e))
            if any(
                isinstance(dtype, pl.Null)
                for dtype in pl.datatypes.unpack_dtypes(*polars_schema.values())
            ):
                error = NotImplementedError(
                    f"No GPU support for {result} with Null column dtype."
                )
                self.errors.append(error)
                return ir.ErrorNode(schema, str(error))

            return result

    def translate_expr(self, *, n: int) -> expr.Expr:
        """
        Translate a polars-internal expression IR into our representation.

        Parameters
        ----------
        n
            Node to translate, an integer referencing a polars internal node.

        Returns
        -------
        Translated IR object.

        Notes
        -----
        Any expression nodes that cannot be translated are replaced by
        :class:`expr.ErrorExpr` nodes and collected in the the `errors` attribute.
        After translation is complete, this list of errors should be inspected
        to determine if the query is supported.
        """
        node = self.visitor.view_expression(n)
        dtype = dtypes.from_polars(self.visitor.get_dtype(n))
        try:
            return _translate_expr(node, self, dtype)
        except Exception as e:
            self.errors.append(e)
            return expr.ErrorExpr(dtype, str(e))


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
    node: Any, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    raise NotImplementedError(
        f"Translation for {type(node).__name__}"
    )  # pragma: no cover


@_translate_ir.register
def _(
    node: pl_ir.PythonScan, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    scan_fn, with_columns, source_type, predicate, nrows = node.options
    options = (scan_fn, with_columns, source_type, nrows)
    predicate = (
        translate_named_expr(translator, n=predicate) if predicate is not None else None
    )
    return ir.PythonScan(schema, options, predicate)


@_translate_ir.register
def _(
    node: pl_ir.Scan, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    typ, *options = node.scan_type
    if typ == "ndjson":
        (reader_options,) = map(json.loads, options)
        cloud_options = None
    else:
        reader_options, cloud_options = map(json.loads, options)
    file_options = node.file_options
    with_columns = file_options.with_columns
    n_rows = file_options.n_rows
    if n_rows is None:
        n_rows = -1  # All rows
        skip_rows = 0  # Don't skip
    else:
        # TODO: with versioning, rename on the rust side
        skip_rows, n_rows = n_rows

    row_index = file_options.row_index
    return ir.Scan(
        schema,
        typ,
        reader_options,
        cloud_options,
        translator.config.config.copy(),
        node.paths,
        with_columns,
        skip_rows,
        n_rows,
        row_index,
        translate_named_expr(translator, n=node.predicate)
        if node.predicate is not None
        else None,
    )


@_translate_ir.register
def _(
    node: pl_ir.Cache, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.Cache(schema, node.id_, translator.translate_ir(n=node.input))


@_translate_ir.register
def _(
    node: pl_ir.DataFrameScan, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.DataFrameScan(
        schema,
        node.df,
        node.projection,
        translate_named_expr(translator, n=node.selection)
        if node.selection is not None
        else None,
        translator.config.config.copy(),
    )


@_translate_ir.register
def _(
    node: pl_ir.Select, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        exprs = [translate_named_expr(translator, n=e) for e in node.expr]
    return ir.Select(schema, exprs, node.should_broadcast, inp)


@_translate_ir.register
def _(
    node: pl_ir.GroupBy, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        aggs = [translate_named_expr(translator, n=e) for e in node.aggs]
        keys = [translate_named_expr(translator, n=e) for e in node.keys]
    return ir.GroupBy(
        schema,
        keys,
        aggs,
        node.maintain_order,
        node.options,
        inp,
    )


@_translate_ir.register
def _(
    node: pl_ir.Join, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    # Join key dtypes are dependent on the schema of the left and
    # right inputs, so these must be translated with the relevant
    # input active.
    with set_node(translator.visitor, node.input_left):
        inp_left = translator.translate_ir(n=None)
        left_on = [translate_named_expr(translator, n=e) for e in node.left_on]
    with set_node(translator.visitor, node.input_right):
        inp_right = translator.translate_ir(n=None)
        right_on = [translate_named_expr(translator, n=e) for e in node.right_on]
    if (how := node.options[0]) in {
        "inner",
        "left",
        "right",
        "full",
        "cross",
        "semi",
        "anti",
    }:
        return ir.Join(schema, left_on, right_on, node.options, inp_left, inp_right)
    else:
        how, op1, op2 = how
        if how != "ie_join":
            raise NotImplementedError(
                f"Unsupported join type {how}"
            )  # pragma: no cover; asof joins not yet exposed
        if op2 is None:
            ops = [op1]
        else:
            ops = [op1, op2]

        dtype = plc.DataType(plc.TypeId.BOOL8)
        predicate = functools.reduce(
            functools.partial(
                expr.BinOp, dtype, plc.binaryop.BinaryOperator.LOGICAL_AND
            ),
            (
                expr.BinOp(
                    dtype,
                    expr.BinOp._MAPPING[op],
                    insert_colrefs(
                        left.value,
                        table_ref=plc.expressions.TableReference.LEFT,
                        name_to_index={
                            name: i for i, name in enumerate(inp_left.schema)
                        },
                    ),
                    insert_colrefs(
                        right.value,
                        table_ref=plc.expressions.TableReference.RIGHT,
                        name_to_index={
                            name: i for i, name in enumerate(inp_right.schema)
                        },
                    ),
                )
                for op, left, right in zip(ops, left_on, right_on, strict=True)
            ),
        )

        return ir.ConditionalJoin(schema, predicate, node.options, inp_left, inp_right)


@_translate_ir.register
def _(
    node: pl_ir.HStack, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        exprs = [translate_named_expr(translator, n=e) for e in node.exprs]
    return ir.HStack(schema, exprs, node.should_broadcast, inp)


@_translate_ir.register
def _(
    node: pl_ir.Reduce, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:  # pragma: no cover; polars doesn't emit this node yet
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        exprs = [translate_named_expr(translator, n=e) for e in node.expr]
    return ir.Reduce(schema, exprs, inp)


@_translate_ir.register
def _(
    node: pl_ir.Distinct, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    (keep, subset, maintain_order, zlice) = node.options
    keep = ir.Distinct._KEEP_MAP[keep]
    subset = frozenset(subset) if subset is not None else None
    return ir.Distinct(
        schema,
        keep,
        subset,
        zlice,
        maintain_order,
        translator.translate_ir(n=node.input),
    )


@_translate_ir.register
def _(
    node: pl_ir.Sort, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        by = [translate_named_expr(translator, n=e) for e in node.by_column]
    stable, nulls_last, descending = node.sort_options
    order, null_order = sorting.sort_order(
        descending, nulls_last=nulls_last, num_keys=len(by)
    )
    return ir.Sort(schema, by, order, null_order, stable, node.slice, inp)


@_translate_ir.register
def _(
    node: pl_ir.Slice, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.Slice(
        schema, node.offset, node.len, translator.translate_ir(n=node.input)
    )


@_translate_ir.register
def _(
    node: pl_ir.Filter, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        mask = translate_named_expr(translator, n=node.predicate)
    return ir.Filter(schema, mask, inp)


@_translate_ir.register
def _(
    node: pl_ir.SimpleProjection,
    translator: Translator,
    schema: dict[str, plc.DataType],
) -> ir.IR:
    return ir.Projection(schema, translator.translate_ir(n=node.input))


@_translate_ir.register
def _(
    node: pl_ir.MapFunction, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    name, *options = node.function
    return ir.MapFunction(
        schema,
        name,
        options,
        # TODO: merge_sorted breaks this pattern
        translator.translate_ir(n=node.input),
    )


@_translate_ir.register
def _(
    node: pl_ir.Union, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.Union(
        schema, node.options, *(translator.translate_ir(n=n) for n in node.inputs)
    )


@_translate_ir.register
def _(
    node: pl_ir.HConcat, translator: Translator, schema: dict[str, plc.DataType]
) -> ir.IR:
    return ir.HConcat(schema, *(translator.translate_ir(n=n) for n in node.inputs))


def translate_named_expr(
    translator: Translator, *, n: pl_expr.PyExprIR
) -> expr.NamedExpr:
    """
    Translate a polars-internal named expression IR object into our representation.

    Parameters
    ----------
    translator
        Translator object
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
    return expr.NamedExpr(n.output_name, translator.translate_expr(n=n.node))


@singledispatch
def _translate_expr(
    node: Any, translator: Translator, dtype: plc.DataType
) -> expr.Expr:
    raise NotImplementedError(
        f"Translation for {type(node).__name__}"
    )  # pragma: no cover


@_translate_expr.register
def _(node: pl_expr.Function, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    name, *options = node.function_data
    options = tuple(options)
    if isinstance(name, pl_expr.StringFunction):
        if name in {
            pl_expr.StringFunction.StripChars,
            pl_expr.StringFunction.StripCharsStart,
            pl_expr.StringFunction.StripCharsEnd,
        }:
            column, chars = (translator.translate_expr(n=n) for n in node.input)
            if isinstance(chars, expr.Literal):
                if chars.value == pa.scalar(""):
                    # No-op in polars, but libcudf uses empty string
                    # as signifier to remove whitespace.
                    return column
                elif chars.value == pa.scalar(None):
                    # Polars uses None to mean "strip all whitespace"
                    chars = expr.Literal(
                        column.dtype,
                        pa.scalar("", type=plc.interop.to_arrow(column.dtype)),
                    )
            return expr.StringFunction(
                dtype,
                expr.StringFunction.Name.from_polars(name),
                options,
                column,
                chars,
            )
        return expr.StringFunction(
            dtype,
            expr.StringFunction.Name.from_polars(name),
            options,
            *(translator.translate_expr(n=n) for n in node.input),
        )
    elif isinstance(name, pl_expr.BooleanFunction):
        if name == pl_expr.BooleanFunction.IsBetween:
            column, lo, hi = (translator.translate_expr(n=n) for n in node.input)
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
            expr.BooleanFunction.Name.from_polars(name),
            options,
            *(translator.translate_expr(n=n) for n in node.input),
        )
    elif isinstance(name, pl_expr.TemporalFunction):
        # functions for which evaluation of the expression may not return
        # the same dtype as polars, either due to libcudf returning a different
        # dtype, or due to our internal processing affecting what libcudf returns
        needs_cast = {
            pl_expr.TemporalFunction.Year,
            pl_expr.TemporalFunction.Month,
            pl_expr.TemporalFunction.Day,
            pl_expr.TemporalFunction.WeekDay,
            pl_expr.TemporalFunction.Hour,
            pl_expr.TemporalFunction.Minute,
            pl_expr.TemporalFunction.Second,
            pl_expr.TemporalFunction.Millisecond,
        }
        result_expr = expr.TemporalFunction(
            dtype,
            expr.TemporalFunction.Name.from_polars(name),
            options,
            *(translator.translate_expr(n=n) for n in node.input),
        )
        if name in needs_cast:
            return expr.Cast(dtype, result_expr)
        return result_expr

    elif isinstance(name, str):
        children = (translator.translate_expr(n=n) for n in node.input)
        if name == "log":
            (base,) = options
            (child,) = children
            return expr.BinOp(
                dtype,
                plc.binaryop.BinaryOperator.LOG_BASE,
                child,
                expr.Literal(dtype, pa.scalar(base, type=plc.interop.to_arrow(dtype))),
            )
        elif name == "pow":
            return expr.BinOp(dtype, plc.binaryop.BinaryOperator.POW, *children)
        return expr.UnaryFunction(dtype, name, options, *children)
    raise NotImplementedError(
        f"No handler for Expr function node with {name=}"
    )  # pragma: no cover; polars raises on the rust side for now


@_translate_expr.register
def _(node: pl_expr.Window, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    # TODO: raise in groupby?
    if isinstance(node.options, pl_expr.RollingGroupOptions):
        # pl.col("a").rolling(...)
        return expr.RollingWindow(
            dtype, node.options, translator.translate_expr(n=node.function)
        )
    elif isinstance(node.options, pl_expr.WindowMapping):
        # pl.col("a").over(...)
        return expr.GroupedRollingWindow(
            dtype,
            node.options,
            translator.translate_expr(n=node.function),
            *(translator.translate_expr(n=n) for n in node.partition_by),
        )
    assert_never(node.options)


@_translate_expr.register
def _(node: pl_expr.Literal, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    if isinstance(node.value, plrs.PySeries):
        return expr.LiteralColumn(dtype, pl.Series._from_pyseries(node.value))
    value = pa.scalar(node.value, type=plc.interop.to_arrow(dtype))
    return expr.Literal(dtype, value)


@_translate_expr.register
def _(node: pl_expr.Sort, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    # TODO: raise in groupby
    return expr.Sort(dtype, node.options, translator.translate_expr(n=node.expr))


@_translate_expr.register
def _(node: pl_expr.SortBy, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    options = node.sort_options
    return expr.SortBy(
        dtype,
        (options[0], tuple(options[1]), tuple(options[2])),
        translator.translate_expr(n=node.expr),
        *(translator.translate_expr(n=n) for n in node.by),
    )


@_translate_expr.register
def _(node: pl_expr.Gather, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    return expr.Gather(
        dtype,
        translator.translate_expr(n=node.expr),
        translator.translate_expr(n=node.idx),
    )


@_translate_expr.register
def _(node: pl_expr.Filter, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    return expr.Filter(
        dtype,
        translator.translate_expr(n=node.input),
        translator.translate_expr(n=node.by),
    )


@_translate_expr.register
def _(node: pl_expr.Cast, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    inner = translator.translate_expr(n=node.expr)
    # Push casts into literals so we can handle Cast(Literal(Null))
    if isinstance(inner, expr.Literal):
        return expr.Literal(dtype, inner.value.cast(plc.interop.to_arrow(dtype)))
    elif isinstance(inner, expr.Cast):
        # Translation of Len/Count-agg put in a cast, remove double
        # casts if we have one.
        (inner,) = inner.children
    return expr.Cast(dtype, inner)


@_translate_expr.register
def _(node: pl_expr.Column, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    return expr.Col(dtype, node.name)


@_translate_expr.register
def _(node: pl_expr.Agg, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    value = expr.Agg(
        dtype,
        node.name,
        node.options,
        *(translator.translate_expr(n=n) for n in node.arguments),
    )
    if value.name == "count" and value.dtype.id() != plc.TypeId.INT32:
        return expr.Cast(value.dtype, value)
    return value


@_translate_expr.register
def _(node: pl_expr.Ternary, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    return expr.Ternary(
        dtype,
        translator.translate_expr(n=node.predicate),
        translator.translate_expr(n=node.truthy),
        translator.translate_expr(n=node.falsy),
    )


@_translate_expr.register
def _(
    node: pl_expr.BinaryExpr, translator: Translator, dtype: plc.DataType
) -> expr.Expr:
    return expr.BinOp(
        dtype,
        expr.BinOp._MAPPING[node.op],
        translator.translate_expr(n=node.left),
        translator.translate_expr(n=node.right),
    )


@_translate_expr.register
def _(node: pl_expr.Len, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    value = expr.Len(dtype)
    if dtype.id() != plc.TypeId.INT32:
        return expr.Cast(dtype, value)
    return value  # pragma: no cover; never reached since polars len has uint32 dtype
