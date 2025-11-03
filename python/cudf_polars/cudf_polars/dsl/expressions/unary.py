# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
"""DSL nodes for unary operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from typing_extensions import assert_never

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.utils import dtypes

if TYPE_CHECKING:
    from cudf_polars.containers import DataFrame, DataType

__all__ = ["Cast", "Len", "UnaryFunction"]


class Cast(Expr):
    """Class representing a cast of an expression."""

    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(self, dtype: DataType, value: Expr) -> None:
        self.dtype = dtype
        self.children = (value,)
        self.is_pointwise = True
        if not dtypes.can_cast(value.dtype.plc_type, self.dtype.plc_type):
            raise NotImplementedError(
                f"Can't cast {value.dtype.id().name} to {self.dtype.id().name}"
            )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context)
        return column.astype(self.dtype, stream=df.stream)


class Len(Expr):
    """Class representing the length of an expression."""

    def __init__(self, dtype: DataType) -> None:
        self.dtype = dtype
        self.children = ()
        self.is_pointwise = False

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return Column(
            plc.Column.from_scalar(
                plc.Scalar.from_py(df.num_rows, self.dtype.plc_type, stream=df.stream),
                1,
                stream=df.stream,
            ),
            dtype=self.dtype,
        )

    @property
    def agg_request(self) -> plc.aggregation.Aggregation:  # noqa: D102
        return plc.aggregation.count(plc.types.NullPolicy.INCLUDE)


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
        "negate": plc.unary.UnaryOperator.NEGATE,
    }
    _supported_misc_fns = frozenset(
        {
            "as_struct",
            "drop_nulls",
            "fill_null",
            "fill_null_with_strategy",
            "mask_nans",
            "null_count",
            "rank",
            "round",
            "set_sorted",
            "top_k",
            "unique",
            "value_counts",
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
        self, dtype: DataType, name: str, options: tuple[Any, ...], *children: Expr
    ) -> None:
        self.dtype = dtype
        self.name = name
        self.options = options
        self.children = children
        self.is_pointwise = self.name not in (
            "as_struct",
            "cum_max",
            "cum_min",
            "cum_prod",
            "cum_sum",
            "drop_nulls",
            "rank",
            "top_k",
            "unique",
        )

        if self.name not in UnaryFunction._supported_fns:
            raise NotImplementedError(f"Unary function {name=}")
        if self.name in UnaryFunction._supported_cum_aggs:
            (reverse,) = self.options
            if reverse:
                raise NotImplementedError(
                    "reverse=True is not supported for cumulative aggregations"
                )
        if self.name == "fill_null_with_strategy" and self.options[1] not in {0, None}:
            raise NotImplementedError(
                "Filling null values with limit specified is not yet supported."
            )
        if self.name == "rank":
            method, _, _ = self.options
            if method not in {"average", "min", "max", "dense", "ordinal"}:
                raise NotImplementedError(
                    f"ranking with {method=} is not yet supported"
                )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name == "mask_nans":
            (child,) = self.children
            return child.evaluate(df, context=context).mask_nans(stream=df.stream)
        if self.name == "null_count":
            (column,) = (child.evaluate(df, context=context) for child in self.children)
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        column.null_count, self.dtype.plc_type, stream=df.stream
                    ),
                    1,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        arg: plc.Column | plc.Scalar
        if self.name == "round":
            (
                decimal_places,
                round_mode,
            ) = self.options
            (values,) = (child.evaluate(df, context=context) for child in self.children)
            return Column(
                plc.round.round(
                    values.obj,
                    decimal_places,
                    (
                        plc.round.RoundingMethod.HALF_EVEN
                        if round_mode == "half_to_even"
                        else plc.round.RoundingMethod.HALF_UP
                    ),
                    stream=df.stream,
                ),
                dtype=self.dtype,
            ).sorted_like(values)  # pragma: no cover
        elif self.name == "unique":
            (maintain_order,) = self.options
            (values,) = (child.evaluate(df, context=context) for child in self.children)
            # Only one column, so keep_any is the same as keep_first
            # for stable distinct
            keep = plc.stream_compaction.DuplicateKeepOption.KEEP_ANY
            if values.is_sorted:
                maintain_order = True
                (compacted,) = plc.stream_compaction.unique(
                    plc.Table([values.obj]),
                    [0],
                    keep,
                    plc.types.NullEquality.EQUAL,
                    stream=df.stream,
                ).columns()
            else:
                distinct = (
                    plc.stream_compaction.stable_distinct
                    if maintain_order
                    else plc.stream_compaction.distinct
                )
                (compacted,) = distinct(
                    plc.Table([values.obj]),
                    [0],
                    keep,
                    plc.types.NullEquality.EQUAL,
                    plc.types.NanEquality.ALL_EQUAL,
                    stream=df.stream,
                ).columns()
            column = Column(compacted, dtype=self.dtype)
            if maintain_order:
                column = column.sorted_like(values)
            return column
        elif self.name == "set_sorted":
            (column,) = (child.evaluate(df, context=context) for child in self.children)
            (asc,) = self.options
            order = (
                plc.types.Order.ASCENDING
                if asc == "ascending"
                else plc.types.Order.DESCENDING
            )
            null_order = plc.types.NullOrder.BEFORE
            if column.null_count > 0 and (n := column.size) > 1:
                # PERF: This invokes four stream synchronisations!
                has_nulls_first = not plc.copying.get_element(
                    column.obj, 0, stream=df.stream
                ).is_valid()
                has_nulls_last = not plc.copying.get_element(
                    column.obj, n - 1, stream=df.stream
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
        elif self.name == "value_counts":
            (sort, _, _, normalize) = self.options
            count_agg = [plc.aggregation.count(plc.types.NullPolicy.INCLUDE)]
            gb_requests = [
                plc.groupby.GroupByRequest(
                    child.evaluate(df, context=context).obj, count_agg
                )
                for child in self.children
            ]
            (keys_table, (counts_table,)) = plc.groupby.GroupBy(
                df.table, null_handling=plc.types.NullPolicy.INCLUDE
            ).aggregate(gb_requests)
            if sort:
                sort_indices = plc.sorting.stable_sorted_order(
                    counts_table,
                    [plc.types.Order.DESCENDING],
                    [plc.types.NullOrder.BEFORE],
                    stream=df.stream,
                )
                counts_table = plc.copying.gather(
                    counts_table,
                    sort_indices,
                    plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                    stream=df.stream,
                )
                keys_table = plc.copying.gather(
                    keys_table,
                    sort_indices,
                    plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                    stream=df.stream,
                )
            keys_col = keys_table.columns()[0]
            counts_col = counts_table.columns()[0]
            if normalize:
                total_counts = plc.reduce.reduce(
                    counts_col,
                    plc.aggregation.sum(),
                    plc.DataType(plc.TypeId.UINT64),
                    stream=df.stream,
                )
                counts_col = plc.binaryop.binary_operation(
                    counts_col,
                    total_counts,
                    plc.binaryop.BinaryOperator.DIV,
                    plc.DataType(plc.TypeId.FLOAT64),
                    stream=df.stream,
                )
            elif counts_col.type().id() == plc.TypeId.INT32:
                counts_col = plc.unary.cast(
                    counts_col, plc.DataType(plc.TypeId.UINT32), stream=df.stream
                )

            plc_column = plc.Column(
                self.dtype.plc_type,
                counts_col.size(),
                None,
                None,
                0,
                0,
                [keys_col, counts_col],
            )
            return Column(plc_column, dtype=self.dtype)
        elif self.name == "drop_nulls":
            (column,) = (child.evaluate(df, context=context) for child in self.children)
            if column.null_count == 0:
                return column
            return Column(
                plc.stream_compaction.drop_nulls(
                    plc.Table([column.obj]), [0], 1, stream=df.stream
                ).columns()[0],
                dtype=self.dtype,
            )
        elif self.name == "fill_null":
            column = self.children[0].evaluate(df, context=context)
            if column.null_count == 0:
                return column
            fill_value = self.children[1]
            if isinstance(fill_value, Literal):
                arg = plc.Scalar.from_py(
                    fill_value.value, fill_value.dtype.plc_type, stream=df.stream
                )
            else:
                evaluated = fill_value.evaluate(df, context=context)
                arg = (
                    evaluated.obj_scalar(stream=df.stream)
                    if evaluated.is_scalar
                    else evaluated.obj
                )
            if isinstance(arg, plc.Scalar) and dtypes.can_cast(
                column.dtype.plc_type, arg.type()
            ):  # pragma: no cover
                arg = (
                    Column(
                        plc.Column.from_scalar(arg, 1, stream=df.stream),
                        dtype=fill_value.dtype,
                    )
                    .astype(column.dtype, stream=df.stream)
                    .obj.to_scalar(stream=df.stream)
                )
            return Column(
                plc.replace.replace_nulls(column.obj, arg, stream=df.stream),
                dtype=self.dtype,
            )
        elif self.name == "fill_null_with_strategy":
            column = self.children[0].evaluate(df, context=context)
            strategy, limit = self.options
            if (
                column.null_count == 0
                or limit == 0
                or (
                    column.null_count == column.size and strategy not in {"zero", "one"}
                )
            ):
                return column

            replacement: plc.replace.ReplacePolicy | plc.Scalar
            if strategy == "forward":
                replacement = plc.replace.ReplacePolicy.PRECEDING
            elif strategy == "backward":
                replacement = plc.replace.ReplacePolicy.FOLLOWING
            elif strategy == "min":
                replacement = plc.reduce.reduce(
                    column.obj,
                    plc.aggregation.min(),
                    column.dtype.plc_type,
                    stream=df.stream,
                )
            elif strategy == "max":
                replacement = plc.reduce.reduce(
                    column.obj,
                    plc.aggregation.max(),
                    column.dtype.plc_type,
                    stream=df.stream,
                )
            elif strategy == "mean":
                replacement = plc.reduce.reduce(
                    column.obj,
                    plc.aggregation.mean(),
                    plc.DataType(plc.TypeId.FLOAT64),
                    stream=df.stream,
                )
            elif strategy == "zero":
                replacement = plc.scalar.Scalar.from_py(
                    0, dtype=column.dtype.plc_type, stream=df.stream
                )
            elif strategy == "one":
                replacement = plc.scalar.Scalar.from_py(
                    1, dtype=column.dtype.plc_type, stream=df.stream
                )
            else:
                assert_never(strategy)  # pragma: no cover

            if strategy == "mean":
                return Column(
                    plc.replace.replace_nulls(
                        plc.unary.cast(
                            column.obj,
                            plc.DataType(plc.TypeId.FLOAT64),
                            stream=df.stream,
                        ),
                        replacement,
                        stream=df.stream,
                    ),
                    dtype=self.dtype,
                ).astype(self.dtype, stream=df.stream)
            return Column(
                plc.replace.replace_nulls(column.obj, replacement, stream=df.stream),
                dtype=self.dtype,
            )
        elif self.name == "as_struct":
            children = [
                child.evaluate(df, context=context).obj for child in self.children
            ]
            return Column(
                plc.Column(
                    data_type=self.dtype.plc_type,
                    size=children[0].size(),
                    data=None,
                    mask=None,
                    null_count=0,
                    offset=0,
                    children=children,
                ),
                dtype=self.dtype,
            )
        elif self.name == "rank":
            (column,) = (child.evaluate(df, context=context) for child in self.children)
            method_str, descending, _ = self.options

            method = {
                "average": plc.aggregation.RankMethod.AVERAGE,
                "min": plc.aggregation.RankMethod.MIN,
                "max": plc.aggregation.RankMethod.MAX,
                "dense": plc.aggregation.RankMethod.DENSE,
                "ordinal": plc.aggregation.RankMethod.FIRST,
            }[method_str]

            order = (
                plc.types.Order.DESCENDING if descending else plc.types.Order.ASCENDING
            )

            ranked: plc.Column = plc.sorting.rank(
                column.obj,
                method,
                order,
                plc.types.NullPolicy.EXCLUDE,
                plc.types.NullOrder.BEFORE if descending else plc.types.NullOrder.AFTER,
                percentage=False,
                stream=df.stream,
            )

            # Min/Max/Dense/Ordinal -> IDX_DTYPE
            # See https://github.com/pola-rs/polars/blob/main/crates/polars-ops/src/series/ops/rank.rs
            if method_str in {"min", "max", "dense", "ordinal"}:
                dest = self.dtype.plc_type.id()
                src = ranked.type().id()
                if dest == plc.TypeId.UINT32 and src != plc.TypeId.UINT32:
                    ranked = plc.unary.cast(
                        ranked, plc.DataType(plc.TypeId.UINT32), stream=df.stream
                    )
                elif (
                    dest == plc.TypeId.UINT64 and src != plc.TypeId.UINT64
                ):  # pragma: no cover
                    ranked = plc.unary.cast(
                        ranked, plc.DataType(plc.TypeId.UINT64), stream=df.stream
                    )

            return Column(ranked, dtype=self.dtype)
        elif self.name == "top_k":
            (column, k) = (
                child.evaluate(df, context=context) for child in self.children
            )
            (reverse,) = self.options
            return Column(
                plc.sorting.top_k(
                    column.obj,
                    cast(Literal, self.children[1]).value,
                    plc.types.Order.ASCENDING
                    if reverse
                    else plc.types.Order.DESCENDING,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name in self._OP_MAPPING:
            column = self.children[0].evaluate(df, context=context)
            if column.dtype.plc_type.id() != self.dtype.id():
                arg = plc.unary.cast(column.obj, self.dtype.plc_type, stream=df.stream)
            else:
                arg = column.obj
            return Column(
                plc.unary.unary_operation(
                    arg, self._OP_MAPPING[self.name], stream=df.stream
                ),
                dtype=self.dtype,
            )
        elif self.name in UnaryFunction._supported_cum_aggs:
            column = self.children[0].evaluate(df, context=context)
            plc_col = column.obj
            col_type = column.dtype.plc_type
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
                    plc.TypeId.INT8,
                    plc.TypeId.UINT8,
                    plc.TypeId.INT16,
                    plc.TypeId.UINT16,
                }
            ) or (
                self.name == "cum_prod"
                and plc.traits.is_integral(col_type)
                and plc.types.size_of(col_type) <= 4
            ):
                plc_col = plc.unary.cast(
                    plc_col, plc.DataType(plc.TypeId.INT64), stream=df.stream
                )
            elif (
                self.name == "cum_sum"
                and column.dtype.plc_type.id() == plc.TypeId.BOOL8
            ):
                plc_col = plc.unary.cast(
                    plc_col, plc.DataType(plc.TypeId.UINT32), stream=df.stream
                )
            if self.name == "cum_sum":
                agg = plc.aggregation.sum()
            elif self.name == "cum_prod":
                agg = plc.aggregation.product()
            elif self.name == "cum_min":
                agg = plc.aggregation.min()
            elif self.name == "cum_max":
                agg = plc.aggregation.max()

            return Column(
                plc.reduce.scan(
                    plc_col, agg, plc.reduce.ScanType.INCLUSIVE, stream=df.stream
                ),
                dtype=self.dtype,
            )
        raise NotImplementedError(
            f"Unimplemented unary function {self.name=}"
        )  # pragma: no cover; init trips first
