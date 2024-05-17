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
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import pyarrow as pa

from polars.polars import _expr_nodes as pl_expr

import cudf._lib.pylibcudf as plc

from cudf_polars.containers import Column, Scalar
from cudf_polars.utils import sorting

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    "RollingWindow",
    "GroupedRollingWindow",
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


class Expr:
    __slots__ = ("dtype", "hash_value", "repr_value")
    #: Data type of the expression
    dtype: plc.DataType
    #: caching slot for the hash of the expression
    hash_value: int
    #: caching slot for repr of the expression
    repr_value: str
    #: Children of the expression
    children: tuple[Expr, ...] = ()
    #: Names of non-child data (not Exprs) for reconstruction
    _non_child: ClassVar[tuple[str, ...]] = ("dtype",)

    # Constructor must take arguments in order (*_non_child, *children)
    def __init__(self, dtype: plc.DataType) -> None:
        self.dtype = dtype

    def _ctor_arguments(self, children: Sequence[Expr]) -> Sequence:
        return (*(getattr(self, attr) for attr in self._non_child), *children)

    def get_hash(self) -> int:
        """
        Return the hash of this expr.

        Override this in subclasses, rather than __hash__.

        Returns
        -------
        The integer hash value.
        """
        return hash((type(self), self._ctor_arguments(self.children)))

    def __hash__(self):
        """Hash of an expression with caching."""
        try:
            return self.hash_value
        except AttributeError:
            self.hash_value = self.get_hash()
            return self.hash_value

    def is_equal(self, other: Any) -> bool:
        """
        Equality of two expressions.

        Override this in subclasses, rather than __eq__.

        Parameter
        ---------
        other
            object to compare to

        Returns
        -------
        True if the two expressions are equal, false otherwise.
        """
        if type(self) is not type(other):
            return False
        return self._ctor_arguments(self.children) == other._ctor_arguments(
            other.children
        )

    def __eq__(self, other):
        """Equality of expressions."""
        if type(self) != type(other) or hash(self) != hash(other):
            return False
        else:
            return self.is_equal(other)

    def __ne__(self, other):
        """Inequality of expressions."""
        return not self.__eq__(other)

    def __repr__(self):
        """String representation of an expression with caching."""
        try:
            return self.repr_value
        except AttributeError:
            args = ", ".join(f"{arg!r}" for arg in self._ctor_arguments(self.children))
            self.repr_value = f"{type(self).__name__}({args})"
            return self.repr_value

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
    ) -> Column:
        """Look up self in the mapping before evaluating it."""
        if mapping is None:
            return fn(self, df, context=context, mapping=mapping)
        else:
            try:
                return mapping[self]
            except KeyError:
                return fn(self, df, context=context, mapping=mapping)

    return look


class NamedExpr(Expr):
    __slots__ = ("name", "children")
    _non_child = ("dtype", "name")

    def __init__(self, dtype: plc.DataType, name: str, value: Expr) -> None:
        super().__init__(dtype)
        self.name = name
        self.children = (value,)

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        return Column(
            child.evaluate(df, context=context, mapping=mapping).obj, self.name
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        (value,) = self.children
        return value.collect_agg(depth=depth)


class Literal(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: pa.Scalar

    def __init__(self, dtype: plc.DataType, value: Any) -> None:
        super().__init__(dtype)
        self.value = pa.scalar(value)

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
        obj = plc.interop.from_arrow(self.value)
        return Scalar(obj)  # type: ignore

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        raise NotImplementedError("Literal in groupby")


class Col(Expr):
    __slots__ = ("name",)
    _non_child = ("dtype", "name")
    name: str

    def __init__(self, dtype: plc.DataType, name: str) -> None:
        self.dtype = dtype
        self.name = name

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


class BooleanFunction(Expr):
    __slots__ = ("name", "options", "children")
    _non_child = ("dtype", "name", "options")

    def __init__(self, dtype: plc.DataType, name: str, options: Any, *children: Expr):
        super().__init__(dtype)
        self.options = options
        self.name = name
        self.children = tuple(children)


class Sort(Expr):
    __slots__ = ("options", "children")
    _non_child = ("dtype", "options")

    def __init__(
        self, dtype: plc.DataType, options: tuple[bool, bool, bool], column: Expr
    ):
        super().__init__(dtype)
        self.options = options
        self.children = (column,)

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context, mapping=mapping)
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


class SortBy(Expr):
    __slots__ = ("options", "children")
    _non_child = ("dtype", "options")

    def __init__(
        self,
        dtype: plc.DataType,
        options: tuple[bool, bool, tuple[bool]],
        column: Expr,
        *by: Expr,
    ):
        super().__init__(dtype)
        self.options = options
        self.children = (column, *by)

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        column, *by = (
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        )
        (stable, nulls_last, descending) = self.options
        order, null_order = sorting.sort_order(
            descending, nulls_last=nulls_last, num_keys=len(by)
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


class Gather(Expr):
    __slots__ = ("children",)
    _non_child = ("dtype",)

    def __init__(self, dtype: plc.DataType, values: Expr, indices: Expr):
        super().__init__(dtype)
        self.children = (values, indices)

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        values, indices = (
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        )
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


class Filter(Expr):
    __slots__ = ("children",)
    _non_child = ("dtype",)

    def __init__(self, dtype: plc.DataType, values: Expr, indices: Expr):
        super().__init__(dtype)
        self.children = (values, indices)

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        values, mask = (
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        )
        table = plc.stream_compaction.apply_boolean_mask(
            plc.Table([values.obj]), mask.obj
        )
        return Column(table.columns()[0], values.name).with_sorted(like=values)

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: Could do with sort-based groupby and segmented filter
        raise NotImplementedError("Filter in groupby")


class RollingWindow(Expr):
    __slots__ = ("options", "children")
    _non_child = ("dtype", "options")

    def __init__(self, dtype: plc.DataType, options: Any, agg: Expr):
        super().__init__(dtype)
        self.options = options
        self.children = (agg,)


class GroupedRollingWindow(Expr):
    __slots__ = ("options", "children")
    _non_child = ("dtype", "options")

    def __init__(self, dtype: plc.DataType, options: Any, agg: Expr, *by: Expr):
        super().__init__(dtype)
        self.options = options
        self.children = (agg, *by)


class Cast(Expr):
    __slots__ = ("children",)
    _non_child = ("dtype",)

    def __init__(self, dtype: plc.DataType, value: Expr):
        super().__init__(dtype)
        self.children = (value,)

    @with_mapping
    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: dict[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context, mapping=mapping)
        return Column(plc.unary.cast(column.obj, self.dtype), column.name).with_sorted(
            like=column
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: Could do with sort-based groupby and segmented filter
        (child,) = self.children
        return child.collect_agg(depth=depth)


class Agg(Expr):
    __slots__ = ("name", "options", "op", "request", "children")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self, dtype: plc.DataType, name: str, options: Any, value: Expr
    ) -> None:
        super().__init__(dtype)
        self.name = name
        self.options = options
        self.children = (value,)
        if name not in Agg._SUPPORTED:
            raise NotImplementedError(f"Unsupported aggregation {name=}")
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

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        if depth >= 1:
            raise NotImplementedError("Nested aggregations in groupby")
        (child,) = self.children
        ((expr, _, _),) = child.collect_agg(depth=depth + 1).requests
        if self.request is None:
            raise NotImplementedError(f"Aggregation {self.name} in groupby")
        return AggInfo([(expr, self.request, self)])

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
        (child,) = self.children
        return self.op(child.evaluate(df, context=context, mapping=mapping))


class BinOp(Expr):
    __slots__ = ("op", "children")
    _non_child = ("dtype", "op")

    def __init__(
        self,
        dtype: plc.DataType,
        op: plc.binaryop.BinaryOperator,
        left: Expr,
        right: Expr,
    ) -> None:
        super().__init__(dtype)
        self.op = op
        self.children = (left, right)

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
        left, right = (
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        )
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
            left_info, right_info = (
                child.collect_agg(depth=depth) for child in self.children
            )
            return AggInfo(
                [*left_info.requests, *right_info.requests],
            )
