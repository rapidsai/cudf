# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""
DSL nodes for the polars expression language.

An expression node is a function, `DataFrame -> Column`.

The evaluation context is provided by a LogicalPlan node, and can
affect the evaluation rule as well as providing the dataframe input.
In particular, the interpretation of the expression language in a
`GroupBy` node is groupwise, rather than whole frame.
"""

from __future__ import annotations

import enum
from enum import IntEnum
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import pyarrow as pa

from polars.polars import _expr_nodes as pl_expr

import cudf._lib.pylibcudf as plc

from cudf_polars.containers import Column, NamedColumn
from cudf_polars.utils import sorting

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars.type_aliases as pl_types

    from cudf_polars.containers import DataFrame

__all__ = [
    "Expr",
    "NamedExpr",
    "Literal",
    "Col",
    "BooleanFunction",
    "StringFunction",
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
    """
    An abstract expression object.

    This contains a (potentially empty) tuple of child expressions,
    along with non-child data. For uniform reconstruction and
    implementation of hashing and equality schemes, child classes need
    to provide a certain amount of metadata when they are defined.
    Specifically, the ``_non_child`` attribute must list, in-order,
    the names of the slots that are passed to the constructor. The
    constructor must take arguments in the order ``(*_non_child,
    *children).``
    """

    __slots__ = ("dtype", "_hash_value", "_repr_value")
    dtype: plc.DataType
    """Data type of the expression."""
    _hash_value: int
    """Caching slot for the hash of the expression."""
    _repr_value: str
    """Caching slot for repr of the expression."""
    children: tuple[Expr, ...] = ()
    """Children of the expression."""
    _non_child: ClassVar[tuple[str, ...]] = ("dtype",)
    """Names of non-child data (not Exprs) for reconstruction."""

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

    def __hash__(self) -> int:
        """Hash of an expression with caching."""
        try:
            return self._hash_value
        except AttributeError:
            self._hash_value = self.get_hash()
            return self._hash_value

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

    def __eq__(self, other: Any) -> bool:
        """Equality of expressions."""
        if type(self) != type(other) or hash(self) != hash(other):
            return False
        else:
            return self.is_equal(other)

    def __ne__(self, other: Any) -> bool:
        """Inequality of expressions."""
        return not self.__eq__(other)

    def __repr__(self) -> str:
        """String representation of an expression with caching."""
        try:
            return self._repr_value
        except AttributeError:
            args = ", ".join(f"{arg!r}" for arg in self._ctor_arguments(self.children))
            self._repr_value = f"{type(self).__name__}({args})"
            return self._repr_value

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """
        Evaluate this expression given a dataframe for context.

        Parameters
        ----------
        df
            DataFrame that will provide columns.
        context
            What context are we performing this evaluation in?
        mapping
            Substitution mapping from expressions to Columns, used to
            override the evaluation of a given expression if we're
            performing a simple rewritten evaluation.

        Notes
        -----
        Do not call this function directly, but rather
        :meth:`evaluate` which handles the mapping lookups.

        Returns
        -------
        Column representing the evaluation of the expression.

        Raises
        ------
        NotImplementedError
            If we couldn't evaluate the expression. Ideally all these
            are returned during translation to the IR, but for now we
            are not perfect.
        """
        raise NotImplementedError(f"Evaluation of {type(self).__name__}")

    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """
        Evaluate this expression given a dataframe for context.

        Parameters
        ----------
        df
            DataFrame that will provide columns.
        context
            What context are we performing this evaluation in?
        mapping
            Substitution mapping from expressions to Columns, used to
            override the evaluation of a given expression if we're
            performing a simple rewritten evaluation.

        Notes
        -----
        Individual subclasses should implement :meth:`do_evaluate`,
        this method provides logic to handle lookups in the
        substitution mapping.

        Returns
        -------
        Column representing the evaluation of the expression.

        Raises
        ------
        NotImplementedError
            If we couldn't evaluate the expression. Ideally all these
            are returned during translation to the IR, but for now we
            are not perfect.
        """
        if mapping is None:
            return self.do_evaluate(df, context=context, mapping=mapping)
        try:
            return mapping[self]
        except KeyError:
            return self.do_evaluate(df, context=context, mapping=mapping)

    def collect_agg(self, *, depth: int) -> AggInfo:
        """
        Collect information about aggregations in groupbys.

        Parameters
        ----------
        depth
            The depth of aggregating (reduction or sampling)
            expressions we are currently at.

        Returns
        -------
        Aggregation info describing the expression to aggregate in the
        groupby.

        Raises
        ------
        NotImplementedError
            If we can't currently perform the aggregation request, for
            example nested aggregations like ``a.max().min()``.
        """
        raise NotImplementedError(
            f"Collecting aggregation info for {type(self).__name__}"
        )


class NamedExpr:
    # NamedExpr does not inherit from Expr since it does not appear
    # when evaluating expressions themselves, only when constructing
    # named return values in dataframe (IR) nodes.
    __slots__ = ("name", "value")
    value: Expr
    name: str

    def __init__(self, name: str, value: Expr) -> None:
        self.name = name
        self.value = value

    def __hash__(self) -> int:
        """Hash of the expression."""
        return hash((type(self), self.name, self.value))

    def __repr__(self) -> str:
        """Repr of the expression."""
        return f"NamedExpr({self.name}, {self.value}"

    def __eq__(self, other: Any) -> bool:
        """Equality of two expressions."""
        return (
            type(self) is type(other)
            and self.name == other.name
            and self.value == other.value
        )

    def __ne__(self, other: Any) -> bool:
        """Inequality of expressions."""
        return not self.__eq__(other)

    def evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> NamedColumn:
        """
        Evaluate this expression given a dataframe for context.

        Parameters
        ----------
        df
            DataFrame providing context
        context
            Execution context
        mapping
            Substitution mapping

        Returns
        -------
        NamedColumn attaching a name to an evaluated Column

        See Also
        --------
        :meth:`Expr.evaluate` for details, this function just adds the
        name to a column produced from an expression.
        """
        obj = self.value.evaluate(df, context=context, mapping=mapping)
        return NamedColumn(
            obj.obj,
            self.name,
            is_sorted=obj.is_sorted,
            order=obj.order,
            null_order=obj.null_order,
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return self.value.collect_agg(depth=depth)


class Literal(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: pa.Scalar[Any]
    children: tuple[()]

    def __init__(self, dtype: plc.DataType, value: pa.Scalar[Any]) -> None:
        super().__init__(dtype)
        assert value.type == plc.interop.to_arrow(dtype)
        self.value = value

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        # datatype of pyarrow scalar is correct by construction.
        return Column(plc.Column.from_scalar(plc.interop.from_arrow(self.value), 1))


class Col(Expr):
    __slots__ = ("name",)
    _non_child = ("dtype", "name")
    name: str
    children: tuple[()]

    def __init__(self, dtype: plc.DataType, name: str) -> None:
        self.dtype = dtype
        self.name = name

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return df._column_map[self.name]

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return AggInfo([(self, plc.aggregation.collect_list(), self)])


class Len(Expr):
    children: tuple[()]

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


class BooleanFunction(Expr):
    __slots__ = ("name", "options", "children")
    _non_child = ("dtype", "name", "options")
    children: tuple[Expr, ...]

    def __init__(
        self,
        dtype: plc.DataType,
        name: pl_expr.BooleanFunction,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        super().__init__(dtype)
        self.options = options
        self.name = name
        self.children = children
        if (
            self.name in (pl_expr.BooleanFunction.Any, pl_expr.BooleanFunction.All)
            and not self.options[0]
        ):
            # With ignore_nulls == False, polars uses Kleene logic
            raise NotImplementedError(f"Kleene logic for {self.name}")
        if self.name in (
            pl_expr.BooleanFunction.IsFinite,
            pl_expr.BooleanFunction.IsInfinite,
            pl_expr.BooleanFunction.IsIn,
        ):
            raise NotImplementedError(f"{self.name}")

    @staticmethod
    def _distinct(
        column: Column,
        *,
        keep: plc.stream_compaction.DuplicateKeepOption,
        source_value: plc.Scalar,
        target_value: plc.Scalar,
    ) -> Column:
        table = plc.Table([column.obj])
        indices = plc.stream_compaction.distinct_indices(
            table,
            keep,
            # TODO: polars doesn't expose options for these
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
        )
        return Column(
            plc.copying.scatter(
                [source_value],
                indices,
                plc.Table([plc.Column.from_scalar(target_value, table.num_rows())]),
            ).columns()[0]
        )

    _BETWEEN_OPS: ClassVar[
        dict[
            pl_types.ClosedInterval,
            tuple[plc.binaryop.BinaryOperator, plc.binaryop.BinaryOperator],
        ]
    ] = {
        "none": (
            plc.binaryop.BinaryOperator.GREATER,
            plc.binaryop.BinaryOperator.LESS,
        ),
        "left": (
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            plc.binaryop.BinaryOperator.LESS,
        ),
        "right": (
            plc.binaryop.BinaryOperator.GREATER,
            plc.binaryop.BinaryOperator.LESS_EQUAL,
        ),
        "both": (
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            plc.binaryop.BinaryOperator.LESS_EQUAL,
        ),
    }

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = [
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        ]
        if self.name == pl_expr.BooleanFunction.Any:
            (column,) = columns
            return plc.Column.from_scalar(
                plc.reduce.reduce(column.obj, plc.aggregation.any(), self.dtype), 1
            )
        elif self.name == pl_expr.BooleanFunction.All:
            (column,) = columns
            return plc.Column.from_scalar(
                plc.reduce.reduce(column.obj, plc.aggregation.all(), self.dtype), 1
            )
        if self.name == pl_expr.BooleanFunction.IsNull:
            (column,) = columns
            return Column(plc.unary.is_null(column.obj))
        elif self.name == pl_expr.BooleanFunction.IsNotNull:
            (column,) = columns
            return Column(plc.unary.is_valid(column.obj))
        elif self.name == pl_expr.BooleanFunction.IsNan:
            # TODO: copy over null mask since is_nan(null) => null in polars
            (column,) = columns
            return Column(plc.unary.is_nan(column.obj))
        elif self.name == pl_expr.BooleanFunction.IsNotNan:
            # TODO: copy over null mask since is_not_nan(null) => null in polars
            (column,) = columns
            return Column(plc.unary.is_not_nan(column.obj))
        elif self.name == pl_expr.BooleanFunction.IsFirstDistinct:
            (column,) = columns
            return self._distinct(
                column,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
                source_value=plc.interop.from_arrow(
                    pa.scalar(value=True, type=plc.interop.to_arrow(self.dtype))
                ),
                target_value=plc.interop.from_arrow(
                    pa.scalar(value=False, type=plc.interop.to_arrow(self.dtype))
                ),
            )
        elif self.name == pl_expr.BooleanFunction.IsLastDistinct:
            (column,) = columns
            return self._distinct(
                column,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
                source_value=plc.interop.from_arrow(
                    pa.scalar(value=True, type=plc.interop.to_arrow(self.dtype))
                ),
                target_value=plc.interop.from_arrow(
                    pa.scalar(value=False, type=plc.interop.to_arrow(self.dtype))
                ),
            )
        elif self.name == pl_expr.BooleanFunction.IsUnique:
            (column,) = columns
            return self._distinct(
                column,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
                source_value=plc.interop.from_arrow(
                    pa.scalar(value=True, type=plc.interop.to_arrow(self.dtype))
                ),
                target_value=plc.interop.from_arrow(
                    pa.scalar(value=False, type=plc.interop.to_arrow(self.dtype))
                ),
            )
        elif self.name == pl_expr.BooleanFunction.IsDuplicated:
            (column,) = columns
            return self._distinct(
                column,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
                source_value=plc.interop.from_arrow(
                    pa.scalar(value=False, type=plc.interop.to_arrow(self.dtype))
                ),
                target_value=plc.interop.from_arrow(
                    pa.scalar(value=True, type=plc.interop.to_arrow(self.dtype))
                ),
            )
        elif self.name == pl_expr.BooleanFunction.AllHorizontal:
            if any(c.obj.null_count() > 0 for c in columns):
                raise NotImplementedError("Kleene logic for all_horizontal")
            return Column(
                reduce(
                    partial(
                        plc.binaryop.binary_operation,
                        op=plc.binaryop.BinaryOperator.BITWISE_AND,
                        output_type=self.dtype,
                    ),
                    (c.obj for c in columns),
                )
            )
        elif self.name == pl_expr.BooleanFunction.AnyHorizontal:
            if any(c.obj.null_count() > 0 for c in columns):
                raise NotImplementedError("Kleene logic for any_horizontal")
            return Column(
                reduce(
                    partial(
                        plc.binaryop.binary_operation,
                        op=plc.binaryop.BinaryOperator.BITWISE_OR,
                        output_type=self.dtype,
                    ),
                    (c.obj for c in columns),
                )
            )
        elif self.name == pl_expr.BooleanFunction.IsBetween:
            column, lo, hi = columns
            (closed,) = self.options
            lop, rop = self._BETWEEN_OPS[closed]
            return Column(
                plc.binaryop.binary_operation(
                    plc.binaryop.binary_operation(
                        column.obj, lo.obj, lop, output_type=self.dtype
                    ),
                    plc.binaryop.binary_operation(
                        column.obj, hi.obj, rop, output_type=self.dtype
                    ),
                    plc.binaryop.BinaryOperator.LOGICAL_AND,
                    self.dtype,
                )
            )
        else:
            raise NotImplementedError(f"BooleanFunction {self.name}")


class StringFunction(Expr):
    __slots__ = ("name", "options", "children")
    _non_child = ("dtype", "name", "options")
    children: tuple[Expr, ...]

    def __init__(
        self,
        dtype: plc.DataType,
        name: pl_expr.StringFunction,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        super().__init__(dtype)
        self.options = options
        self.name = name
        self.children = children
        if self.name not in (
            pl_expr.StringFunction.Lowercase,
            pl_expr.StringFunction.Uppercase,
            pl_expr.StringFunction.EndsWith,
            pl_expr.StringFunction.StartsWith,
        ):
            raise NotImplementedError(f"String function {self.name}")

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = [
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        ]
        if self.name == pl_expr.StringFunction.Lowercase:
            (column,) = columns
            return Column(plc.strings.case.to_lower(column.obj))
        elif self.name == pl_expr.StringFunction.Uppercase:
            (column,) = columns
            return Column(plc.strings.case.to_upper(column.obj))
        elif self.name == pl_expr.StringFunction.EndsWith:
            column, suffix = columns
            return Column(
                plc.strings.find.ends_with(
                    column.obj,
                    suffix.obj_scalar
                    if column.obj.size() != suffix.obj.size() and suffix.is_scalar
                    else suffix.obj,
                )
            )
        elif self.name == pl_expr.StringFunction.StartsWith:
            column, prefix = columns
            return Column(
                plc.strings.find.starts_with(
                    column.obj,
                    prefix.obj_scalar
                    if column.obj.size() != prefix.obj.size() and prefix.is_scalar
                    else prefix.obj,
                )
            )
        else:
            raise NotImplementedError(
                f"StringFunction {self.name}"
            )  # pragma: no cover; handled by init raising


class Sort(Expr):
    __slots__ = ("options", "children")
    _non_child = ("dtype", "options")
    children: tuple[Expr]

    def __init__(
        self, dtype: plc.DataType, options: tuple[bool, bool, bool], column: Expr
    ) -> None:
        super().__init__(dtype)
        self.options = options
        self.children = (column,)

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
        (stable, nulls_last, descending) = self.options
        order, null_order = sorting.sort_order(
            [descending], nulls_last=[nulls_last], num_keys=1
        )
        do_sort = plc.sorting.stable_sort if stable else plc.sorting.sort
        table = do_sort(plc.Table([column.obj]), order, null_order)
        return Column(
            table.columns()[0],
            is_sorted=plc.types.Sorted.YES,
            order=order[0],
            null_order=null_order[0],
        )


class SortBy(Expr):
    __slots__ = ("options", "children")
    _non_child = ("dtype", "options")
    children: tuple[Expr, ...]

    def __init__(
        self,
        dtype: plc.DataType,
        options: tuple[bool, tuple[bool], tuple[bool]],
        column: Expr,
        *by: Expr,
    ) -> None:
        super().__init__(dtype)
        self.options = options
        self.children = (column, *by)

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
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
        return Column(table.columns()[0])


class Gather(Expr):
    __slots__ = ("children",)
    _non_child = ("dtype",)
    children: tuple[Expr, Expr]

    def __init__(self, dtype: plc.DataType, values: Expr, indices: Expr) -> None:
        super().__init__(dtype)
        self.children = (values, indices)

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
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
                plc.interop.from_arrow(
                    pa.scalar(n, type=plc.interop.to_arrow(indices.obj.data_type()))
                ),
            )
        else:
            bounds_policy = plc.copying.OutOfBoundsPolicy.DONT_CHECK
            obj = indices.obj
        table = plc.copying.gather(plc.Table([values.obj]), obj, bounds_policy)
        return Column(table.columns()[0])


class Filter(Expr):
    __slots__ = ("children",)
    _non_child = ("dtype",)
    children: tuple[Expr, Expr]

    def __init__(self, dtype: plc.DataType, values: Expr, indices: Expr):
        super().__init__(dtype)
        self.children = (values, indices)

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        values, mask = (
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        )
        table = plc.stream_compaction.apply_boolean_mask(
            plc.Table([values.obj]), mask.obj
        )
        return Column(table.columns()[0]).sorted_like(values)


class RollingWindow(Expr):
    __slots__ = ("options", "children")
    _non_child = ("dtype", "options")
    children: tuple[Expr]

    def __init__(self, dtype: plc.DataType, options: Any, agg: Expr) -> None:
        super().__init__(dtype)
        self.options = options
        self.children = (agg,)


class GroupedRollingWindow(Expr):
    __slots__ = ("options", "children")
    _non_child = ("dtype", "options")
    children: tuple[Expr, ...]

    def __init__(self, dtype: plc.DataType, options: Any, agg: Expr, *by: Expr) -> None:
        super().__init__(dtype)
        self.options = options
        self.children = (agg, *by)


class Cast(Expr):
    __slots__ = ("children",)
    _non_child = ("dtype",)
    children: tuple[Expr]

    def __init__(self, dtype: plc.DataType, value: Expr) -> None:
        super().__init__(dtype)
        self.children = (value,)

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
        return Column(plc.unary.cast(column.obj, self.dtype)).sorted_like(column)

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: Could do with sort-based groupby and segmented filter
        (child,) = self.children
        return child.collect_agg(depth=depth)


class Agg(Expr):
    __slots__ = ("name", "options", "op", "request", "children")
    _non_child = ("dtype", "name", "options")
    children: tuple[Expr]

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
        elif name == "n_unique":
            # TODO: datatype of result
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
            req = plc.aggregation.count(null_handling=plc.types.NullPolicy.EXCLUDE)
        else:
            raise NotImplementedError
        self.request = req
        op = getattr(self, f"_{name}", None)
        if op is None:
            op = partial(self._reduce, request=req)
        elif name in {"min", "max"}:
            op = partial(op, propagate_nans=options)
        elif name in {"count", "first", "last"}:
            pass
        else:
            raise AssertionError
        self.op = op

    _SUPPORTED: ClassVar[frozenset[str]] = frozenset(
        [
            "min",
            "max",
            "median",
            "n_unique",
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
            is_sorted=plc.types.Sorted.YES,
            order=plc.types.Order.ASCENDING,
            null_order=plc.types.NullOrder.BEFORE,
        )

    def _count(self, column: Column) -> Column:
        return Column(
            plc.Column.from_scalar(
                plc.interop.from_arrow(
                    pa.scalar(
                        column.obj.size() - column.obj.null_count(),
                        type=plc.interop.to_arrow(self.dtype),
                    ),
                ),
                1,
            ),
            is_sorted=plc.types.Sorted.YES,
            order=plc.types.Order.ASCENDING,
            null_order=plc.types.NullOrder.BEFORE,
        )

    def _min(self, column: Column, *, propagate_nans: bool) -> Column:
        if propagate_nans and column.nan_count > 0:
            return Column(
                plc.Column.from_scalar(
                    plc.interop.from_arrow(
                        pa.scalar(float("nan"), type=plc.interop.to_arrow(self.dtype))
                    ),
                    1,
                ),
                is_sorted=plc.types.Sorted.YES,
                order=plc.types.Order.ASCENDING,
                null_order=plc.types.NullOrder.BEFORE,
            )
        if column.nan_count > 0:
            column = column.mask_nans()
        return self._reduce(column, request=plc.aggregation.min())

    def _max(self, column: Column, *, propagate_nans: bool) -> Column:
        if propagate_nans and column.nan_count > 0:
            return Column(
                plc.Column.from_scalar(
                    plc.interop.from_arrow(
                        pa.scalar(float("nan"), type=plc.interop.to_arrow(self.dtype))
                    ),
                    1,
                ),
                is_sorted=plc.types.Sorted.YES,
                order=plc.types.Order.ASCENDING,
                null_order=plc.types.NullOrder.BEFORE,
            )
        if column.nan_count > 0:
            column = column.mask_nans()
        return self._reduce(column, request=plc.aggregation.max())

    def _first(self, column: Column) -> Column:
        return Column(
            plc.copying.slice(column.obj, [0, 1])[0],
            is_sorted=plc.types.Sorted.YES,
            order=plc.types.Order.ASCENDING,
            null_order=plc.types.NullOrder.BEFORE,
        )

    def _last(self, column: Column) -> Column:
        n = column.obj.size()
        return Column(
            plc.copying.slice(column.obj, [n - 1, n])[0],
            is_sorted=plc.types.Sorted.YES,
            order=plc.types.Order.ASCENDING,
            null_order=plc.types.NullOrder.BEFORE,
        )

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if context is not ExecutionContext.FRAME:
            raise NotImplementedError(f"Agg in context {context}")
        (child,) = self.children
        return self.op(child.evaluate(df, context=context, mapping=mapping))


class BinOp(Expr):
    __slots__ = ("op", "children")
    _non_child = ("dtype", "op")
    children: tuple[Expr, Expr]

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

    _MAPPING: ClassVar[dict[pl_expr.Operator, plc.binaryop.BinaryOperator]] = {
        pl_expr.Operator.Eq: plc.binaryop.BinaryOperator.EQUAL,
        pl_expr.Operator.EqValidity: plc.binaryop.BinaryOperator.NULL_EQUALS,
        pl_expr.Operator.NotEq: plc.binaryop.BinaryOperator.NOT_EQUAL,
        pl_expr.Operator.NotEqValidity: plc.binaryop.BinaryOperator.NULL_NOT_EQUALS,
        pl_expr.Operator.Lt: plc.binaryop.BinaryOperator.LESS,
        pl_expr.Operator.LtEq: plc.binaryop.BinaryOperator.LESS_EQUAL,
        pl_expr.Operator.Gt: plc.binaryop.BinaryOperator.GREATER,
        pl_expr.Operator.GtEq: plc.binaryop.BinaryOperator.GREATER_EQUAL,
        pl_expr.Operator.Plus: plc.binaryop.BinaryOperator.ADD,
        pl_expr.Operator.Minus: plc.binaryop.BinaryOperator.SUB,
        pl_expr.Operator.Multiply: plc.binaryop.BinaryOperator.MUL,
        pl_expr.Operator.Divide: plc.binaryop.BinaryOperator.DIV,
        pl_expr.Operator.TrueDivide: plc.binaryop.BinaryOperator.TRUE_DIV,
        pl_expr.Operator.FloorDivide: plc.binaryop.BinaryOperator.FLOOR_DIV,
        pl_expr.Operator.Modulus: plc.binaryop.BinaryOperator.PYMOD,
        pl_expr.Operator.And: plc.binaryop.BinaryOperator.BITWISE_AND,
        pl_expr.Operator.Or: plc.binaryop.BinaryOperator.BITWISE_OR,
        pl_expr.Operator.Xor: plc.binaryop.BinaryOperator.BITWISE_XOR,
        pl_expr.Operator.LogicalAnd: plc.binaryop.BinaryOperator.LOGICAL_AND,
        pl_expr.Operator.LogicalOr: plc.binaryop.BinaryOperator.LOGICAL_OR,
    }

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        left, right = (
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        )
        lop = left.obj
        rop = right.obj
        if left.obj.size() != right.obj.size():
            if left.is_scalar:
                lop = left.obj_scalar
            elif right.is_scalar:
                rop = right.obj_scalar
        return Column(
            plc.binaryop.binary_operation(lop, rop, self.op, self.dtype),
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        if depth == 1:
            # inside aggregation, need to pre-evaluate,
            # groupby construction has checked that we don't have
            # nested aggs, so stop the recursion and return ourselves
            # for pre-eval
            return AggInfo([(self, plc.aggregation.collect_list(), self)])
        else:
            left_info, right_info = (
                child.collect_agg(depth=depth) for child in self.children
            )
            requests = [*left_info.requests, *right_info.requests]
            # TODO: Hack, if there were no reductions inside this
            # binary expression then we want to pre-evaluate and
            # collect ourselves. Otherwise we want to collect the
            # aggregations inside and post-evaluate. This is a bad way
            # of checking that we are in case 1.
            if all(
                agg.kind() == plc.aggregation.Kind.COLLECT_LIST
                for _, agg, _ in requests
            ):
                return AggInfo([(self, plc.aggregation.collect_list(), self)])
            return AggInfo(
                [*left_info.requests, *right_info.requests],
            )
