# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Translate polars IR representation to ours."""

from __future__ import annotations

import functools
import json
from contextlib import AbstractContextManager, nullcontext
from functools import singledispatch
from typing import TYPE_CHECKING, Any

from typing_extensions import assert_never

import polars as pl
import polars.polars as plrs
from polars.polars import _expr_nodes as pl_expr, _ir_nodes as pl_ir

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.to_ast import insert_colrefs
from cudf_polars.dsl.utils.aggregations import decompose_single_agg
from cudf_polars.dsl.utils.groupby import rewrite_groupby
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.dsl.utils.replace import replace
from cudf_polars.dsl.utils.rolling import rewrite_rolling
from cudf_polars.typing import Schema
from cudf_polars.utils import config, sorting

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
    engine
        GPU engine configuration.
    """

    def __init__(self, visitor: NodeTraverser, engine: GPUEngine):
        self.visitor = visitor
        self.config_options = config.ConfigOptions.from_polars_engine(engine)
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
        if (version := self.visitor.version()) >= (7, 1):
            e = NotImplementedError(
                f"No support for polars IR {version=}"
            )  # pragma: no cover; no such version for now.
            self.errors.append(e)  # pragma: no cover
            raise e  # pragma: no cover

        with ctx:
            polars_schema = self.visitor.get_schema()
            try:
                schema = {k: DataType(v) for k, v in polars_schema.items()}
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

    def translate_expr(self, *, n: int, schema: Schema) -> expr.Expr:
        """
        Translate a polars-internal expression IR into our representation.

        Parameters
        ----------
        n
            Node to translate, an integer referencing a polars internal node.
        schema
            Schema of the IR node this expression uses as evaluation context.

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
        dtype = DataType(self.visitor.get_dtype(n))
        try:
            return _translate_expr(node, self, dtype, schema)
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
def _translate_ir(node: Any, translator: Translator, schema: Schema) -> ir.IR:
    raise NotImplementedError(
        f"Translation for {type(node).__name__}"
    )  # pragma: no cover


@_translate_ir.register
def _(node: pl_ir.PythonScan, translator: Translator, schema: Schema) -> ir.IR:
    scan_fn, with_columns, source_type, predicate, nrows = node.options
    options = (scan_fn, with_columns, source_type, nrows)
    predicate = (
        translate_named_expr(translator, n=predicate, schema=schema)
        if predicate is not None
        else None
    )
    return ir.PythonScan(schema, options, predicate)


@_translate_ir.register
def _(node: pl_ir.Scan, translator: Translator, schema: Schema) -> ir.IR:
    typ, *options = node.scan_type
    if typ == "ndjson":
        (reader_options,) = map(json.loads, options)
        cloud_options = None
    else:
        reader_options, cloud_options = map(json.loads, options)
    file_options = node.file_options
    with_columns = file_options.with_columns
    row_index = file_options.row_index
    include_file_paths = file_options.include_file_paths
    config_options = translator.config_options
    parquet_options = config_options.parquet_options

    pre_slice = file_options.n_rows
    if pre_slice is None:
        n_rows = -1
        skip_rows = 0
    else:
        skip_rows, n_rows = pre_slice

    return ir.Scan(
        schema,
        typ,
        reader_options,
        cloud_options,
        node.paths,
        with_columns,
        skip_rows,
        n_rows,
        row_index,
        include_file_paths,
        translate_named_expr(translator, n=node.predicate, schema=schema)
        if node.predicate is not None
        else None,
        parquet_options,
    )


@_translate_ir.register
def _(node: pl_ir.Cache, translator: Translator, schema: Schema) -> ir.IR:
    return ir.Cache(
        schema, node.id_, node.cache_hits, translator.translate_ir(n=node.input)
    )


@_translate_ir.register
def _(node: pl_ir.DataFrameScan, translator: Translator, schema: Schema) -> ir.IR:
    return ir.DataFrameScan(
        schema,
        node.df,
        node.projection,
    )


@_translate_ir.register
def _(node: pl_ir.Select, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        exprs = [
            translate_named_expr(translator, n=e, schema=inp.schema) for e in node.expr
        ]
    return ir.Select(schema, exprs, node.should_broadcast, inp)


@_translate_ir.register
def _(node: pl_ir.GroupBy, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        keys = [
            translate_named_expr(translator, n=e, schema=inp.schema) for e in node.keys
        ]
        original_aggs = [
            translate_named_expr(translator, n=e, schema=inp.schema) for e in node.aggs
        ]
    is_rolling = node.options.rolling is not None
    is_dynamic = node.options.dynamic is not None
    if is_dynamic:
        raise NotImplementedError("group_by_dynamic")
    elif is_rolling:
        return rewrite_rolling(
            node.options, schema, keys, original_aggs, translator.config_options, inp
        )
    else:
        return rewrite_groupby(node, schema, keys, original_aggs, inp)


@_translate_ir.register
def _(node: pl_ir.Join, translator: Translator, schema: Schema) -> ir.IR:
    # Join key dtypes are dependent on the schema of the left and
    # right inputs, so these must be translated with the relevant
    # input active.
    with set_node(translator.visitor, node.input_left):
        inp_left = translator.translate_ir(n=None)
        left_on = [
            translate_named_expr(translator, n=e, schema=inp_left.schema)
            for e in node.left_on
        ]
    with set_node(translator.visitor, node.input_right):
        inp_right = translator.translate_ir(n=None)
        right_on = [
            translate_named_expr(translator, n=e, schema=inp_right.schema)
            for e in node.right_on
        ]

    if (how := node.options[0]) in {
        "Inner",
        "Left",
        "Right",
        "Full",
        "Cross",
        "Semi",
        "Anti",
    }:
        return ir.Join(
            schema,
            left_on,
            right_on,
            node.options,
            inp_left,
            inp_right,
        )
    else:
        how, op1, op2 = node.options[0]
        if how != "IEJoin":
            raise NotImplementedError(
                f"Unsupported join type {how}"
            )  # pragma: no cover; asof joins not yet exposed
        if op2 is None:
            ops = [op1]
        else:
            ops = [op1, op2]

        dtype = DataType(pl.datatypes.Boolean())
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
def _(node: pl_ir.HStack, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        exprs = [
            translate_named_expr(translator, n=e, schema=inp.schema) for e in node.exprs
        ]
    return ir.HStack(schema, exprs, node.should_broadcast, inp)


@_translate_ir.register
def _(
    node: pl_ir.Reduce, translator: Translator, schema: Schema
) -> ir.IR:  # pragma: no cover; polars doesn't emit this node yet
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        exprs = [
            translate_named_expr(translator, n=e, schema=inp.schema) for e in node.expr
        ]
    return ir.Reduce(schema, exprs, inp)


@_translate_ir.register
def _(node: pl_ir.Distinct, translator: Translator, schema: Schema) -> ir.IR:
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
def _(node: pl_ir.Sort, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        by = [
            translate_named_expr(translator, n=e, schema=inp.schema)
            for e in node.by_column
        ]
    stable, nulls_last, descending = node.sort_options
    order, null_order = sorting.sort_order(
        descending, nulls_last=nulls_last, num_keys=len(by)
    )
    return ir.Sort(schema, by, order, null_order, stable, node.slice, inp)


@_translate_ir.register
def _(node: pl_ir.Slice, translator: Translator, schema: Schema) -> ir.IR:
    return ir.Slice(
        schema, node.offset, node.len, translator.translate_ir(n=node.input)
    )


@_translate_ir.register
def _(node: pl_ir.Filter, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        mask = translate_named_expr(translator, n=node.predicate, schema=inp.schema)
    return ir.Filter(schema, mask, inp)


@_translate_ir.register
def _(node: pl_ir.SimpleProjection, translator: Translator, schema: Schema) -> ir.IR:
    return ir.Projection(schema, translator.translate_ir(n=node.input))


@_translate_ir.register
def _(node: pl_ir.MergeSorted, translator: Translator, schema: Schema) -> ir.IR:
    key = node.key
    inp_left = translator.translate_ir(n=node.input_left)
    inp_right = translator.translate_ir(n=node.input_right)
    return ir.MergeSorted(
        schema,
        key,
        inp_left,
        inp_right,
    )


@_translate_ir.register
def _(node: pl_ir.MapFunction, translator: Translator, schema: Schema) -> ir.IR:
    name, *options = node.function
    return ir.MapFunction(
        schema,
        name,
        options,
        translator.translate_ir(n=node.input),
    )


@_translate_ir.register
def _(node: pl_ir.Union, translator: Translator, schema: Schema) -> ir.IR:
    return ir.Union(
        schema, node.options, *(translator.translate_ir(n=n) for n in node.inputs)
    )


@_translate_ir.register
def _(node: pl_ir.HConcat, translator: Translator, schema: Schema) -> ir.IR:
    return ir.HConcat(
        schema,
        False,  # noqa: FBT003
        *(translator.translate_ir(n=n) for n in node.inputs),
    )


@_translate_ir.register
def _(node: pl_ir.Sink, translator: Translator, schema: Schema) -> ir.IR:
    payload = json.loads(node.payload)
    try:
        file = payload["File"]
        sink_kind_options = file["file_type"]
    except KeyError as err:  # pragma: no cover
        raise NotImplementedError("Unsupported payload structure") from err
    if isinstance(sink_kind_options, dict):
        if len(sink_kind_options) != 1:  # pragma: no cover; not sure if this can happen
            raise NotImplementedError("Sink options dict with more than one entry.")
        sink_kind, options = next(iter(sink_kind_options.items()))
    else:
        raise NotImplementedError(
            "Unsupported sink options structure"
        )  # pragma: no cover

    sink_options = file.get("sink_options", {})
    cloud_options = file.get("cloud_options")

    options.update(sink_options)

    return ir.Sink(
        schema=schema,
        kind=sink_kind,
        path=file["target"],
        options=options,
        cloud_options=cloud_options,
        df=translator.translate_ir(n=node.input),
    )


def translate_named_expr(
    translator: Translator, *, n: pl_expr.PyExprIR, schema: Schema
) -> expr.NamedExpr:
    """
    Translate a polars-internal named expression IR object into our representation.

    Parameters
    ----------
    translator
        Translator object
    n
        Node to translate, a named expression node.
    schema
        Schema of the IR node this expression uses as evaluation context.

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
    return expr.NamedExpr(
        n.output_name, translator.translate_expr(n=n.node, schema=schema)
    )


@singledispatch
def _translate_expr(
    node: Any, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    raise NotImplementedError(
        f"Translation for {type(node).__name__}"
    )  # pragma: no cover


@_translate_expr.register
def _(
    node: pl_expr.Function, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    name, *options = node.function_data
    options = tuple(options)
    if isinstance(name, pl_expr.StringFunction):
        if name in {
            pl_expr.StringFunction.StripChars,
            pl_expr.StringFunction.StripCharsStart,
            pl_expr.StringFunction.StripCharsEnd,
        }:
            column, chars = (
                translator.translate_expr(n=n, schema=schema) for n in node.input
            )
            if isinstance(chars, expr.Literal):
                # We check for null first because we want to use the
                # chars type, but it is invalid to try and
                # produce a string scalar with a null dtype.
                if chars.value is None:
                    # Polars uses None to mean "strip all whitespace"
                    chars = expr.Literal(column.dtype, "")
                elif chars.value == "":
                    # No-op in polars, but libcudf uses empty string
                    # as signifier to remove whitespace.
                    return column
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
            *(translator.translate_expr(n=n, schema=schema) for n in node.input),
        )
    elif isinstance(name, pl_expr.BooleanFunction):
        if name == pl_expr.BooleanFunction.IsBetween:
            column, lo, hi = (
                translator.translate_expr(n=n, schema=schema) for n in node.input
            )
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
            *(translator.translate_expr(n=n, schema=schema) for n in node.input),
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
            *(translator.translate_expr(n=n, schema=schema) for n in node.input),
        )
        if name in needs_cast:
            return expr.Cast(dtype, result_expr)
        return result_expr

    elif isinstance(name, str):
        children = (translator.translate_expr(n=n, schema=schema) for n in node.input)
        if name == "log":
            (base,) = options
            (child,) = children
            return expr.BinOp(
                dtype,
                plc.binaryop.BinaryOperator.LOG_BASE,
                child,
                expr.Literal(dtype, base),
            )
        elif name == "pow":
            return expr.BinOp(dtype, plc.binaryop.BinaryOperator.POW, *children)
        elif name in "top_k":
            (col, k) = children
            assert isinstance(k, expr.Literal)
            (descending,) = options
            return expr.Slice(
                dtype,
                0,
                k.value,
                expr.Sort(dtype, (False, True, not descending), col),
            )

        return expr.UnaryFunction(dtype, name, options, *children)
    raise NotImplementedError(
        f"No handler for Expr function node with {name=}"
    )  # pragma: no cover; polars raises on the rust side for now


@_translate_expr.register
def _(
    node: pl_expr.Window, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    if isinstance(node.options, pl_expr.RollingGroupOptions):
        # pl.col("a").rolling(...)
        agg = translator.translate_expr(n=node.function, schema=schema)
        name_generator = unique_names(schema)
        aggs, named_post_agg = decompose_single_agg(
            expr.NamedExpr(next(name_generator), agg), name_generator, is_top=True
        )
        named_aggs = [agg for agg, _ in aggs]
        orderby = node.options.index_column
        orderby_dtype = schema[orderby].plc
        if plc.traits.is_integral(orderby_dtype):
            # Integer orderby column is cast in implementation to int64 in polars
            orderby_dtype = plc.DataType(plc.TypeId.INT64)
        closed_window = node.options.closed_window
        if isinstance(named_post_agg.value, expr.Col):
            (named_agg,) = named_aggs
            return expr.RollingWindow(
                named_agg.value.dtype,
                orderby_dtype,
                node.options.offset,
                node.options.period,
                closed_window,
                orderby,
                named_agg.value,
            )
        replacements: dict[expr.Expr, expr.Expr] = {
            expr.Col(agg.value.dtype, agg.name): expr.RollingWindow(
                agg.value.dtype,
                orderby_dtype,
                node.options.offset,
                node.options.period,
                closed_window,
                orderby,
                agg.value,
            )
            for agg in named_aggs
        }
        return replace([named_post_agg.value], replacements)[0]
    elif isinstance(node.options, pl_expr.WindowMapping):
        # pl.col("a").over(...)
        return expr.GroupedRollingWindow(
            dtype,
            node.options,
            translator.translate_expr(n=node.function, schema=schema),
            *(translator.translate_expr(n=n, schema=schema) for n in node.partition_by),
        )
    assert_never(node.options)


@_translate_expr.register
def _(
    node: pl_expr.Literal, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    if isinstance(node.value, plrs.PySeries):
        return expr.LiteralColumn(dtype, pl.Series._from_pyseries(node.value))
    if dtype.id() == plc.TypeId.LIST:  # pragma: no cover
        # TODO: Remove once pylibcudf.Scalar supports lists
        return expr.LiteralColumn(dtype, pl.Series(node.value))
    return expr.Literal(dtype, node.value)


@_translate_expr.register
def _(
    node: pl_expr.Sort, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    # TODO: raise in groupby
    return expr.Sort(
        dtype, node.options, translator.translate_expr(n=node.expr, schema=schema)
    )


@_translate_expr.register
def _(
    node: pl_expr.SortBy, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    options = node.sort_options
    return expr.SortBy(
        dtype,
        (options[0], tuple(options[1]), tuple(options[2])),
        translator.translate_expr(n=node.expr, schema=schema),
        *(translator.translate_expr(n=n, schema=schema) for n in node.by),
    )


@_translate_expr.register
def _(
    node: pl_expr.Slice, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    offset = translator.translate_expr(n=node.offset, schema=schema)
    length = translator.translate_expr(n=node.length, schema=schema)
    assert isinstance(offset, expr.Literal)
    assert isinstance(length, expr.Literal)
    return expr.Slice(
        dtype,
        offset.value,
        length.value,
        translator.translate_expr(n=node.input, schema=schema),
    )


@_translate_expr.register
def _(
    node: pl_expr.Gather, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    return expr.Gather(
        dtype,
        translator.translate_expr(n=node.expr, schema=schema),
        translator.translate_expr(n=node.idx, schema=schema),
    )


@_translate_expr.register
def _(
    node: pl_expr.Filter, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    return expr.Filter(
        dtype,
        translator.translate_expr(n=node.input, schema=schema),
        translator.translate_expr(n=node.by, schema=schema),
    )


@_translate_expr.register
def _(
    node: pl_expr.Cast, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    inner = translator.translate_expr(n=node.expr, schema=schema)
    # Push casts into literals so we can handle Cast(Literal(Null))
    if isinstance(inner, expr.Literal):
        return inner.astype(dtype)
    elif isinstance(inner, expr.Cast):
        # Translation of Len/Count-agg put in a cast, remove double
        # casts if we have one.
        (inner,) = inner.children
    return expr.Cast(dtype, inner)


@_translate_expr.register
def _(
    node: pl_expr.Column, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    return expr.Col(dtype, node.name)


@_translate_expr.register
def _(
    node: pl_expr.Agg, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    value = expr.Agg(
        dtype,
        node.name,
        node.options,
        *(translator.translate_expr(n=n, schema=schema) for n in node.arguments),
    )
    if value.name == "count" and value.dtype.id() != plc.TypeId.INT32:
        return expr.Cast(value.dtype, value)
    return value


@_translate_expr.register
def _(
    node: pl_expr.Ternary, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    return expr.Ternary(
        dtype,
        translator.translate_expr(n=node.predicate, schema=schema),
        translator.translate_expr(n=node.truthy, schema=schema),
        translator.translate_expr(n=node.falsy, schema=schema),
    )


@_translate_expr.register
def _(
    node: pl_expr.BinaryExpr,
    translator: Translator,
    dtype: DataType,
    schema: Schema,
) -> expr.Expr:
    return expr.BinOp(
        dtype,
        expr.BinOp._MAPPING[node.op],
        translator.translate_expr(n=node.left, schema=schema),
        translator.translate_expr(n=node.right, schema=schema),
    )


@_translate_expr.register
def _(
    node: pl_expr.Len, translator: Translator, dtype: DataType, schema: Schema
) -> expr.Expr:
    value = expr.Len(dtype)
    if dtype.id() != plc.TypeId.INT32:
        return expr.Cast(dtype, value)
    return value  # pragma: no cover; never reached since polars len has uint32 dtype
