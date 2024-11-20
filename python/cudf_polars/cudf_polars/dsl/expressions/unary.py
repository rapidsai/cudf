# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
"""DSL nodes for unary operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import AggInfo, ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.utils import dtypes

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.containers import DataFrame

__all__ = ["Cast", "UnaryFunction", "Len"]


class Cast(Expr):
    """Class representing a cast of an expression."""

    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(self, dtype: plc.DataType, value: Expr) -> None:
        self.dtype = dtype
        self.children = (value,)
        if not dtypes.can_cast(value.dtype, self.dtype):
            raise NotImplementedError(
                f"Can't cast {value.dtype.id().name} to {self.dtype.id().name}"
            )

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context, mapping=mapping)
        return column.astype(self.dtype)

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: Could do with sort-based groupby and segmented filter
        (child,) = self.children
        return child.collect_agg(depth=depth)


class Len(Expr):
    """Class representing the length of an expression."""

    def __init__(self, dtype: plc.DataType) -> None:
        self.dtype = dtype
        self.children = ()

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return Column(
            plc.Column.from_scalar(
                plc.interop.from_arrow(
                    pa.scalar(df.num_rows, type=plc.interop.to_arrow(self.dtype))
                ),
                1,
            )
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: polars returns a uint, not an int for count
        return AggInfo(
            [(None, plc.aggregation.count(plc.types.NullPolicy.INCLUDE), self)]
        )


class UnaryFunction(Expr):
    """Class representing unary functions of an expression."""

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    # Note: log, and pow are handled via translation to binops
    _OP_MAPPING: ClassVar[dict[str, plc.unary.UnaryOperator]] = {
        "sin": plc.unary.UnaryOperator.SIN,
        "cos": plc.unary.UnaryOperator.COS,
        "tan": plc.unary.UnaryOperator.TAN,
        "arcsin": plc.unary.UnaryOperator.ARCSIN,
        "arccos": plc.unary.UnaryOperator.ARCCOS,
        "arctan": plc.unary.UnaryOperator.ARCTAN,
        "sinh": plc.unary.UnaryOperator.SINH,
        "cosh": plc.unary.UnaryOperator.COSH,
        "tanh": plc.unary.UnaryOperator.TANH,
        "arcsinh": plc.unary.UnaryOperator.ARCSINH,
        "arccosh": plc.unary.UnaryOperator.ARCCOSH,
        "arctanh": plc.unary.UnaryOperator.ARCTANH,
        "exp": plc.unary.UnaryOperator.EXP,
        "sqrt": plc.unary.UnaryOperator.SQRT,
        "cbrt": plc.unary.UnaryOperator.CBRT,
        "ceil": plc.unary.UnaryOperator.CEIL,
        "floor": plc.unary.UnaryOperator.FLOOR,
        "abs": plc.unary.UnaryOperator.ABS,
        "bit_invert": plc.unary.UnaryOperator.BIT_INVERT,
        "not": plc.unary.UnaryOperator.NOT,
    }
    _supported_misc_fns = frozenset(
        {
            "drop_nulls",
            "fill_null",
            "mask_nans",
            "round",
            "set_sorted",
            "unique",
        }
    )
    _supported_cum_aggs = frozenset(
        {
            "cum_min",
            "cum_max",
            "cum_prod",
            "cum_sum",
        }
    )
    _supported_fns = frozenset().union(
        _supported_misc_fns, _supported_cum_aggs, _OP_MAPPING.keys()
    )

    def __init__(
        self, dtype: plc.DataType, name: str, options: tuple[Any, ...], *children: Expr
    ) -> None:
        self.dtype = dtype
        self.name = name
        self.options = options
        self.children = children

        if self.name not in UnaryFunction._supported_fns:
            raise NotImplementedError(f"Unary function {name=}")
        if self.name in UnaryFunction._supported_cum_aggs:
            (reverse,) = self.options
            if reverse:
                raise NotImplementedError(
                    "reverse=True is not supported for cumulative aggregations"
                )

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name == "mask_nans":
            (child,) = self.children
            return child.evaluate(df, context=context, mapping=mapping).mask_nans()
        if self.name == "round":
            (decimal_places,) = self.options
            (values,) = (
                child.evaluate(df, context=context, mapping=mapping)
                for child in self.children
            )
            return Column(
                plc.round.round(
                    values.obj, decimal_places, plc.round.RoundingMethod.HALF_UP
                )
            ).sorted_like(values)
        elif self.name == "unique":
            (maintain_order,) = self.options
            (values,) = (
                child.evaluate(df, context=context, mapping=mapping)
                for child in self.children
            )
            # Only one column, so keep_any is the same as keep_first
            # for stable distinct
            keep = plc.stream_compaction.DuplicateKeepOption.KEEP_ANY
            if values.is_sorted:
                maintain_order = True
                result = plc.stream_compaction.unique(
                    plc.Table([values.obj]),
                    [0],
                    keep,
                    plc.types.NullEquality.EQUAL,
                )
            else:
                distinct = (
                    plc.stream_compaction.stable_distinct
                    if maintain_order
                    else plc.stream_compaction.distinct
                )
                result = distinct(
                    plc.Table([values.obj]),
                    [0],
                    keep,
                    plc.types.NullEquality.EQUAL,
                    plc.types.NanEquality.ALL_EQUAL,
                )
            (column,) = result.columns()
            if maintain_order:
                return Column(column).sorted_like(values)
            return Column(column)
        elif self.name == "set_sorted":
            (column,) = (
                child.evaluate(df, context=context, mapping=mapping)
                for child in self.children
            )
            (asc,) = self.options
            order = (
                plc.types.Order.ASCENDING
                if asc == "ascending"
                else plc.types.Order.DESCENDING
            )
            null_order = plc.types.NullOrder.BEFORE
            if column.obj.null_count() > 0 and (n := column.obj.size()) > 1:
                # PERF: This invokes four stream synchronisations!
                has_nulls_first = not plc.copying.get_element(column.obj, 0).is_valid()
                has_nulls_last = not plc.copying.get_element(
                    column.obj, n - 1
                ).is_valid()
                if (order == plc.types.Order.DESCENDING and has_nulls_first) or (
                    order == plc.types.Order.ASCENDING and has_nulls_last
                ):
                    null_order = plc.types.NullOrder.AFTER
            return column.set_sorted(
                is_sorted=plc.types.Sorted.YES,
                order=order,
                null_order=null_order,
            )
        elif self.name == "drop_nulls":
            (column,) = (
                child.evaluate(df, context=context, mapping=mapping)
                for child in self.children
            )
            return Column(
                plc.stream_compaction.drop_nulls(
                    plc.Table([column.obj]), [0], 1
                ).columns()[0]
            )
        elif self.name == "fill_null":
            column = self.children[0].evaluate(df, context=context, mapping=mapping)
            if isinstance(self.children[1], Literal):
                arg = plc.interop.from_arrow(self.children[1].value)
            else:
                evaluated = self.children[1].evaluate(
                    df, context=context, mapping=mapping
                )
                arg = evaluated.obj_scalar if evaluated.is_scalar else evaluated.obj
            return Column(plc.replace.replace_nulls(column.obj, arg))
        elif self.name in self._OP_MAPPING:
            column = self.children[0].evaluate(df, context=context, mapping=mapping)
            if column.obj.type().id() != self.dtype.id():
                arg = plc.unary.cast(column.obj, self.dtype)
            else:
                arg = column.obj
            return Column(plc.unary.unary_operation(arg, self._OP_MAPPING[self.name]))
        elif self.name in UnaryFunction._supported_cum_aggs:
            column = self.children[0].evaluate(df, context=context, mapping=mapping)
            plc_col = column.obj
            col_type = column.obj.type()
            # cum_sum casts
            # Int8, UInt8, Int16, UInt16 -> Int64 for overflow prevention
            # Bool -> UInt32
            # cum_prod casts integer dtypes < int64 and bool to int64
            # See:
            # https://github.com/pola-rs/polars/blob/main/crates/polars-ops/src/series/ops/cum_agg.rs
            if (
                self.name == "cum_sum"
                and col_type.id()
                in {
                    plc.types.TypeId.INT8,
                    plc.types.TypeId.UINT8,
                    plc.types.TypeId.INT16,
                    plc.types.TypeId.UINT16,
                }
            ) or (
                self.name == "cum_prod"
                and plc.traits.is_integral(col_type)
                and plc.types.size_of(col_type) <= 4
            ):
                plc_col = plc.unary.cast(
                    plc_col, plc.types.DataType(plc.types.TypeId.INT64)
                )
            elif (
                self.name == "cum_sum"
                and column.obj.type().id() == plc.types.TypeId.BOOL8
            ):
                plc_col = plc.unary.cast(
                    plc_col, plc.types.DataType(plc.types.TypeId.UINT32)
                )
            if self.name == "cum_sum":
                agg = plc.aggregation.sum()
            elif self.name == "cum_prod":
                agg = plc.aggregation.product()
            elif self.name == "cum_min":
                agg = plc.aggregation.min()
            elif self.name == "cum_max":
                agg = plc.aggregation.max()

            return Column(plc.reduce.scan(plc_col, agg, plc.reduce.ScanType.INCLUSIVE))
        raise NotImplementedError(
            f"Unimplemented unary function {self.name=}"
        )  # pragma: no cover; init trips first

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        if self.name in {"unique", "drop_nulls"} | self._supported_cum_aggs:
            raise NotImplementedError(f"{self.name} in groupby")
        if depth == 1:
            # inside aggregation, need to pre-evaluate, groupby
            # construction has checked that we don't have nested aggs,
            # so stop the recursion and return ourselves for pre-eval
            return AggInfo([(self, plc.aggregation.collect_list(), self)])
        else:
            (child,) = self.children
            return child.collect_agg(depth=depth)
