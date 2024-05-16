# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""
DSL nodes for the polars expression language.

An expression node is a function, `DataFrame -> Column` or `DataFrame -> Scalar`.

The evaluation context is provided by a LogicalPlan node, and can
affect the evaluation rule as well as providing the dataframe input.
In particular, the interpretation of the expression language in a
`GroupBy` node is groupwise, rather than whole frame.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import pyarrow as pa

from polars.polars import _expr_nodes as pl_expr

import cudf._lib.pylibcudf as plc

from cudf_polars.containers import Column, Scalar
from cudf_polars.utils import sorting

if TYPE_CHECKING:
    from typing import Callable

    from cudf_polars.containers import DataFrame

__all__ = [
    "Expr",
    "NamedExpr",
    "Literal",
    "Col",
    "BooleanFunction",
    "Sort",
    "SortBy",
    "Gather",
    "Filter",
    "Window",
    "Cast",
    "Agg",
    "BinOp",
]


class ExecutionContext(IntEnum):
    FRAME = enum.auto()
    GROUPBY = enum.auto()
    ROLLING = enum.auto()


class AggInfo(NamedTuple):
    requests: list[tuple[Expr | None, plc.aggregation.Aggregation, Expr]]


@dataclass(slots=True, unsafe_hash=True)
class Expr:
    dtype: plc.DataType

    # TODO: return type is a lie for Literal
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        raise NotImplementedError

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        raise NotImplementedError


def with_mapping(fn):
    """Decorate a callback that takes an expression mapping to use it."""

    def look(
        self,
        df: DataFrame,
        *,
        context=ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ):
        """Look up the self in the mapping before evaluating it."""
        if mapping is None:
            return fn(self, df, context=context, mapping=mapping)
        else:
            try:
                return mapping[self]
            except KeyError:
                return fn(self, df, context=context, mapping=mapping)

    return look


@dataclass(slots=True, unsafe_hash=True)
class NamedExpr(Expr):
    name: str
    value: Expr

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return Column(
            self.value.evaluate(df, context=context, mapping=mapping).obj, self.name
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return self.value.collect_agg(depth=depth)


@dataclass(slots=True, unsafe_hash=True)  # TODO: won't work for list literals
class Literal(Expr):
    value: Any

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        # TODO: obey dtype
        obj = plc.interop.from_arrow(pa.scalar(self.value))
        return Scalar(obj)  # type: ignore

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        raise NotImplementedError("Literal in groupby")


@dataclass(slots=True, unsafe_hash=True)
class Col(Expr):
    name: str

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return df._column_map[self.name]

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return AggInfo([(self, plc.aggregation.collect_list(), self)])


@dataclass(slots=True, unsafe_hash=True)
class Len(Expr):
    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        # TODO: type is wrong, and dtype
        return df.num_rows

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: polars returns a uint, not an int for count
        return AggInfo(
            [(None, plc.aggregation.count(plc.types.NullPolicy.INCLUDE), self)]
        )


@dataclass(slots=True, unsafe_hash=True)
class BooleanFunction(Expr):
    name: str
    options: Any
    arguments: tuple[Expr, ...]


@dataclass(slots=True, unsafe_hash=True)
class Sort(Expr):
    column: Expr
    options: tuple[bool, bool, bool]

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        column = self.column.evaluate(df, context=context, mapping=mapping)
        (stable, nulls_last, descending) = self.options
        order, null_order = sorting.sort_order(
            [descending], nulls_last=nulls_last, num_keys=1
        )
        do_sort = plc.sorting.stable_sort if stable else plc.sorting.sort
        table = do_sort(plc.Table([column.obj]), order, null_order)
        return Column(table.columns()[0], column.name).set_sorted(
            is_sorted=plc.types.Sorted.YES, order=order[0], null_order=null_order[0]
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: Could do with sort-based groupby and segmented sort post-hoc
        raise NotImplementedError("Sort in groupby")


@dataclass(slots=True, unsafe_hash=True)
class SortBy(Expr):
    column: Expr
    by: tuple[Expr, ...]
    options: tuple[bool, bool, tuple[bool]]

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        column = self.column.evaluate(df, context=context, mapping=mapping)
        by = [b.evaluate(df, context=context, mapping=mapping) for b in self.by]
        (stable, nulls_last, descending) = self.options
        order, null_order = sorting.sort_order(
            descending, nulls_last=nulls_last, num_keys=len(self.by)
        )
        do_sort = plc.sorting.stable_sort_by_key if stable else plc.sorting.sort_by_key
        table = do_sort(
            plc.Table([column.obj]), plc.Table([c.obj for c in by]), order, null_order
        )
        return Column(table.columns()[0], column.name)

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: Could do with sort-based groupby and segmented sort post-hoc
        raise NotImplementedError("SortBy in groupby")


@dataclass(slots=True)
class Gather(Expr):
    values: Expr
    indices: Expr

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        values = self.values.evaluate(df, context=context, mapping=mapping)
        indices = self.indices.evaluate(df, context=context, mapping=mapping)
        lo, hi = plc.reduce.minmax(indices.obj)
        lo = plc.interop.to_arrow(lo).as_py()
        hi = plc.interop.to_arrow(hi).as_py()
        n = df.num_rows
        if hi >= n or lo < -n:
            raise ValueError("gather indices are out of bounds")
        if indices.obj.null_count():
            bounds_policy = plc.copying.OutOfBoundsPolicy.NULLIFY
            obj = plc.replace.replace_nulls(
                indices.obj,
                plc.interop.from_arrow(pa.scalar(n), data_type=indices.obj.data_type()),
            )
        else:
            bounds_policy = plc.copying.OutOfBoundsPolicy.DONT_CHECK
            obj = indices.obj
        table = plc.copying.gather(plc.Table([values.obj]), obj, bounds_policy)
        return Column(table.columns()[0], values.name)

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: Could do with sort-based groupby and segmented gather.
        raise NotImplementedError("Gather in groupby")


@dataclass(slots=True)
class Filter(Expr):
    values: Expr
    mask: Expr

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        values = self.values.evaluate(df, context=context, mapping=mapping)
        mask = self.mask.evaluate(df, context=context, mapping=mapping)
        table = plc.stream_compaction.apply_boolean_mask(
            plc.Table([values.obj]), mask.obj
        )
        return Column(table.columns()[0], values.name).with_sorted(like=values)

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: Could do with sort-based groupby and segmented filter
        raise NotImplementedError("Filter in groupby")


@dataclass(slots=True)
class Window(Expr):
    agg: Expr
    by: None | tuple[Expr, ...]
    options: Any


@dataclass(slots=True)
class Cast(Expr):
    dtype: plc.DataType
    column: Expr

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        column = self.column.evaluate(df, context=context, mapping=mapping)
        return Column(plc.unary.cast(column.obj, self.dtype), column.name).with_sorted(
            like=column
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: Could do with sort-based groupby and segmented filter
        return self.column.collect_agg(depth=depth)


@dataclass(slots=True)
class Agg(Expr):
    column: Expr
    op: Callable[..., plc.Column]
    name: str
    request: plc.aggregation.Aggregation

    _SUPPORTED: ClassVar[frozenset[str]] = frozenset(
        [
            "min",
            "max",
            "median",
            "nunique",
            "first",
            "last",
            "mean",
            "sum",
            "count",
            "std",
            "var",
        ]
    )

    def __eq__(self, other):
        """Return whether this Agg is equal to another."""
        return type(self) == type(other) and (self.column, self.name) == (
            other.column,
            other.name,
        )

    def __hash__(self):
        """Return a hash."""
        return hash((self.column, self.name))

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        if depth >= 1:
            raise NotImplementedError("Nested aggregations in groupby")
        ((expr, _, _),) = self.column.collect_agg(depth=depth + 1).requests
        if self.request is None:
            raise NotImplementedError(f"Aggregation {self.name} in groupby")
        return AggInfo([(expr, self.request, self)])

    def __init__(
        self, dtype: plc.DataType, column: Expr, name: str, options: Any
    ) -> None:
        if name not in Agg._SUPPORTED:
            raise NotImplementedError(f"Unsupported aggregation {name}")
        self.dtype = dtype
        self.column = column
        self.name = name
        # TODO: nan handling in groupby case
        if name == "min":
            req = plc.aggregation.min()
        elif name == "max":
            req = plc.aggregation.max()
        elif name == "median":
            req = plc.aggregation.median()
        elif name == "nunique":
            req = plc.aggregation.nunique(null_handling=plc.types.NullPolicy.INCLUDE)
        elif name == "first" or name == "last":
            req = None
        elif name == "mean":
            req = plc.aggregation.mean()
        elif name == "sum":
            req = plc.aggregation.sum()
        elif name == "std":
            # TODO: handle nans
            req = plc.aggregation.std(ddof=options)
        elif name == "var":
            # TODO: handle nans
            req = plc.aggregation.variance(ddof=options)
        elif name == "count":
            req = plc.aggregation.count(null_policy=plc.types.NullPolicy.EXCLUDE)
        else:
            raise NotImplementedError
        self.request = req
        op = getattr(self, f"_{name}", None)
        if op is None:
            op = partial(self._reduce, request=req)
        elif name in {"min", "max"}:
            op = partial(op, propagate_nans=options)
        else:
            raise AssertionError
        self.op = op

    def _reduce(
        self, column: Column, *, request: plc.aggregation.Aggregation
    ) -> Column:
        return Column(
            plc.Column.from_scalar(
                plc.reduce.reduce(column.obj, request, self.dtype),
                1,
            ),
            column.name,
        )

    def _min(self, column: Column, *, propagate_nans: bool) -> Column:
        if propagate_nans and column.nan_count > 0:
            return Column(
                plc.Column.from_scalar(
                    plc.interop.from_arrow(
                        pa.scalar(float("nan")), data_type=self.dtype
                    ),
                    1,
                ),
                column.name,
            )
        if column.nan_count > 0:
            column = column.mask_nans()
        return self._reduce(column, request=plc.aggregation.min())

    def _max(self, column: Column, *, propagate_nans: bool) -> Column:
        if propagate_nans and column.nan_count > 0:
            return Column(
                plc.Column.from_scalar(
                    plc.interop.from_arrow(
                        pa.scalar(float("nan")), data_type=self.dtype
                    ),
                    1,
                ),
                column.name,
            )
        if column.nan_count > 0:
            column = column.mask_nans()
        return self._reduce(column, request=plc.aggregation.max())

    def _first(self, column: Column) -> Column:
        return Column(plc.copying.slice(column.obj, [0, 1])[0], column.name)

    def _last(self, column: Column) -> Column:
        n = column.obj.size()
        return Column(plc.copying.slice(column.obj, [n - 1, n])[0], column.name)

    @with_mapping
    def evaluate(
        self,
        df,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if context is not ExecutionContext.FRAME:
            raise NotImplementedError(f"Agg in context {context}")
        return self.op(self.column.evaluate(df, context=context, mapping=mapping))


@dataclass(slots=True, unsafe_hash=True)
class BinOp(Expr):
    left: Expr
    right: Expr
    op: plc.binaryop.BinaryOperator

    _MAPPING: ClassVar[dict[pl_expr.PyOperator, plc.binaryop.BinaryOperator]] = {
        pl_expr.PyOperator.Eq: plc.binaryop.BinaryOperator.EQUAL,
        pl_expr.PyOperator.EqValidity: plc.binaryop.BinaryOperator.NULL_EQUALS,
        pl_expr.PyOperator.NotEq: plc.binaryop.BinaryOperator.NOT_EQUAL,
        pl_expr.PyOperator.NotEqValidity: plc.binaryop.BinaryOperator.NULL_NOT_EQUALS,
        pl_expr.PyOperator.Lt: plc.binaryop.BinaryOperator.LESS,
        pl_expr.PyOperator.LtEq: plc.binaryop.BinaryOperator.LESS_EQUAL,
        pl_expr.PyOperator.Gt: plc.binaryop.BinaryOperator.GREATER,
        pl_expr.PyOperator.GtEq: plc.binaryop.BinaryOperator.GREATER_EQUAL,
        pl_expr.PyOperator.Plus: plc.binaryop.BinaryOperator.ADD,
        pl_expr.PyOperator.Minus: plc.binaryop.BinaryOperator.SUB,
        pl_expr.PyOperator.Multiply: plc.binaryop.BinaryOperator.MUL,
        pl_expr.PyOperator.Divide: plc.binaryop.BinaryOperator.DIV,
        pl_expr.PyOperator.TrueDivide: plc.binaryop.BinaryOperator.TRUE_DIV,
        pl_expr.PyOperator.FloorDivide: plc.binaryop.BinaryOperator.FLOOR_DIV,
        pl_expr.PyOperator.Modulus: plc.binaryop.BinaryOperator.PYMOD,
        pl_expr.PyOperator.And: plc.binaryop.BinaryOperator.BITWISE_AND,
        pl_expr.PyOperator.Or: plc.binaryop.BinaryOperator.BITWISE_OR,
        pl_expr.PyOperator.Xor: plc.binaryop.BinaryOperator.BITWISE_XOR,
        pl_expr.PyOperator.LogicalAnd: plc.binaryop.BinaryOperator.LOGICAL_AND,
        pl_expr.PyOperator.LogicalOr: plc.binaryop.BinaryOperator.LOGICAL_OR,
    }

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        left = self.left.evaluate(df, context=context, mapping=mapping)
        right = self.right.evaluate(df, context=context, mapping=mapping)
        return Column(
            plc.binaryop.binary_operation(left.obj, right.obj, self.op, self.dtype),
            "what",
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        if depth == 1:
            # inside aggregation, need to pre-evaluate,
            # This recurses to check if we have nested aggs
            # groupby construction has checked that we don't have
            # nested aggs, so stop the recursion and return ourselves
            # for pre-eval
            return AggInfo([(self, plc.aggregation.collect_list(), self)])
        else:
            left_info = self.left.collect_agg(depth=depth)
            right_info = self.right.collect_agg(depth=depth)
            return AggInfo(
                [*left_info.requests, *right_info.requests],
            )
