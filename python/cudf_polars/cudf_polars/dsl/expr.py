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
from cudf_polars.utils import dtypes, sorting

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl
    import polars.type_aliases as pl_types

    from cudf_polars.containers import DataFrame

__all__ = [
    "Expr",
    "NamedExpr",
    "Literal",
    "Col",
    "BooleanFunction",
    "StringFunction",
    "TemporalFunction",
    "Sort",
    "SortBy",
    "Gather",
    "Filter",
    "RollingWindow",
    "GroupedRollingWindow",
    "Cast",
    "Agg",
    "Ternary",
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
            return False  # pragma: no cover; __eq__ trips first
        return self._ctor_arguments(self.children) == other._ctor_arguments(
            other.children
        )

    def __eq__(self, other: Any) -> bool:
        """Equality of expressions."""
        if type(self) is not type(other) or hash(self) != hash(other):
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
        raise NotImplementedError(
            f"Evaluation of expression {type(self).__name__}"
        )  # pragma: no cover; translation of unimplemented nodes trips first

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
        )  # pragma: no cover; check_agg trips first


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
        return f"NamedExpr({self.name}, {self.value})"

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

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return AggInfo([])


class LiteralColumn(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: pa.Array[Any, Any]
    children: tuple[()]

    def __init__(self, dtype: plc.DataType, value: pl.Series) -> None:
        super().__init__(dtype)
        data = value.to_arrow()
        self.value = data.cast(dtypes.downcast_arrow_lists(data.type))

    def get_hash(self) -> int:
        """Compute a hash of the column."""
        # This is stricter than necessary, but we only need this hash
        # for identity in groupby replacements so it's OK. And this
        # way we avoid doing potentially expensive compute.
        return hash((type(self), self.dtype, id(self.value)))

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        # datatype of pyarrow array is correct by construction.
        return Column(plc.interop.from_arrow(self.value))

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return AggInfo([])


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
        if self.name == pl_expr.BooleanFunction.IsIn and not all(
            c.dtype == self.children[0].dtype for c in self.children
        ):
            # TODO: If polars IR doesn't put the casts in, we need to
            # mimic the supertype promotion rules.
            raise NotImplementedError("IsIn doesn't support supertype casting")

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
        if self.name in (
            pl_expr.BooleanFunction.IsFinite,
            pl_expr.BooleanFunction.IsInfinite,
        ):
            # Avoid evaluating the child if the dtype tells us it's unnecessary.
            (child,) = self.children
            is_finite = self.name == pl_expr.BooleanFunction.IsFinite
            if child.dtype.id() not in (plc.TypeId.FLOAT32, plc.TypeId.FLOAT64):
                value = plc.interop.from_arrow(
                    pa.scalar(value=is_finite, type=plc.interop.to_arrow(self.dtype))
                )
                return Column(plc.Column.from_scalar(value, df.num_rows))
            needles = child.evaluate(df, context=context, mapping=mapping)
            to_search = [-float("inf"), float("inf")]
            if is_finite:
                # NaN is neither finite not infinite
                to_search.append(float("nan"))
            haystack = plc.interop.from_arrow(
                pa.array(
                    to_search,
                    type=plc.interop.to_arrow(needles.obj.type()),
                )
            )
            result = plc.search.contains(haystack, needles.obj)
            if is_finite:
                result = plc.unary.unary_operation(result, plc.unary.UnaryOperator.NOT)
            return Column(result)
        columns = [
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        ]
        if self.name == pl_expr.BooleanFunction.Any:
            (column,) = columns
            return Column(
                plc.Column.from_scalar(
                    plc.reduce.reduce(column.obj, plc.aggregation.any(), self.dtype), 1
                )
            )
        elif self.name == pl_expr.BooleanFunction.All:
            (column,) = columns
            return Column(
                plc.Column.from_scalar(
                    plc.reduce.reduce(column.obj, plc.aggregation.all(), self.dtype), 1
                )
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
        elif self.name == pl_expr.BooleanFunction.IsIn:
            needles, haystack = columns
            return Column(plc.search.contains(haystack.obj, needles.obj))
        elif self.name == pl_expr.BooleanFunction.Not:
            (column,) = columns
            return Column(
                plc.unary.unary_operation(column.obj, plc.unary.UnaryOperator.NOT)
            )
        else:
            raise NotImplementedError(
                f"BooleanFunction {self.name}"
            )  # pragma: no cover; handled by init raising


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
        self._validate_input()

    def _validate_input(self):
        if self.name not in (
            pl_expr.StringFunction.Lowercase,
            pl_expr.StringFunction.Uppercase,
            pl_expr.StringFunction.EndsWith,
            pl_expr.StringFunction.StartsWith,
            pl_expr.StringFunction.Contains,
            pl_expr.StringFunction.Slice,
        ):
            raise NotImplementedError(f"String function {self.name}")
        if self.name == pl_expr.StringFunction.Contains:
            literal, strict = self.options
            if not literal:
                if not strict:
                    raise NotImplementedError(
                        "f{strict=} is not supported for regex contains"
                    )
                if not isinstance(self.children[1], Literal):
                    raise NotImplementedError(
                        "Regex contains only supports a scalar pattern"
                    )
        elif self.name == pl_expr.StringFunction.Slice:
            if not all(isinstance(child, Literal) for child in self.children[1:]):
                raise NotImplementedError(
                    "Slice only supports literal start and stop values"
                )

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name == pl_expr.StringFunction.Contains:
            child, arg = self.children
            column = child.evaluate(df, context=context, mapping=mapping)

            literal, _ = self.options
            if literal:
                pat = arg.evaluate(df, context=context, mapping=mapping)
                pattern = (
                    pat.obj_scalar
                    if pat.is_scalar and pat.obj.size() != column.obj.size()
                    else pat.obj
                )
                return Column(plc.strings.find.contains(column.obj, pattern))
            assert isinstance(arg, Literal)
            prog = plc.strings.regex_program.RegexProgram.create(
                arg.value.as_py(),
                flags=plc.strings.regex_flags.RegexFlags.DEFAULT,
            )
            return Column(plc.strings.contains.contains_re(column.obj, prog))
        elif self.name == pl_expr.StringFunction.Slice:
            child, expr_offset, expr_length = self.children
            assert isinstance(expr_offset, Literal)
            assert isinstance(expr_length, Literal)

            column = child.evaluate(df, context=context, mapping=mapping)
            # libcudf slices via [start,stop).
            # polars slices with offset + length where start == offset
            # stop = start + length. Negative values for start look backward
            # from the last element of the string. If the end index would be
            # below zero, an empty string is returned.
            # Do this maths on the host
            start = expr_offset.value.as_py()
            length = expr_length.value.as_py()

            if length == 0:
                stop = start
            else:
                # No length indicates a scan to the end
                # The libcudf equivalent is a null stop
                stop = start + length if length else None
                if length and start < 0 and length >= -start:
                    stop = None
            return Column(
                plc.strings.slice.slice_strings(
                    column.obj,
                    plc.interop.from_arrow(pa.scalar(start, type=pa.int32())),
                    plc.interop.from_arrow(pa.scalar(stop, type=pa.int32())),
                )
            )
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
        raise NotImplementedError(
            f"StringFunction {self.name}"
        )  # pragma: no cover; handled by init raising


class TemporalFunction(Expr):
    __slots__ = ("name", "options", "children")
    _non_child = ("dtype", "name", "options")
    children: tuple[Expr, ...]

    def __init__(
        self,
        dtype: plc.DataType,
        name: pl_expr.TemporalFunction,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        super().__init__(dtype)
        self.options = options
        self.name = name
        self.children = children
        if self.name != pl_expr.TemporalFunction.Year:
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
        if self.name == pl_expr.TemporalFunction.Year:
            (column,) = columns
            return Column(plc.datetime.extract_year(column.obj))
        raise NotImplementedError(
            f"TemporalFunction {self.name}"
        )  # pragma: no cover; init trips first


class UnaryFunction(Expr):
    __slots__ = ("name", "options", "children")
    _non_child = ("dtype", "name", "options")
    children: tuple[Expr, ...]

    def __init__(
        self, dtype: plc.DataType, name: str, options: tuple[Any, ...], *children: Expr
    ) -> None:
        super().__init__(dtype)
        self.name = name
        self.options = options
        self.children = children
        if self.name not in (
            "mask_nans",
            "round",
            "set_sorted",
            "unique",
            "drop_nulls",
            "fill_null",
        ):
            raise NotImplementedError(f"Unary function {name=}")

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

        raise NotImplementedError(
            f"Unimplemented unary function {self.name=}"
        )  # pragma: no cover; init trips first

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        if depth == 1:
            # inside aggregation, need to pre-evaluate, groupby
            # construction has checked that we don't have nested aggs,
            # so stop the recursion and return ourselves for pre-eval
            return AggInfo([(self, plc.aggregation.collect_list(), self)])
        else:
            (child,) = self.children
            return child.collect_agg(depth=depth)


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
                    pa.scalar(n, type=plc.interop.to_arrow(indices.obj.type()))
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
        raise NotImplementedError("Rolling window not implemented")


class GroupedRollingWindow(Expr):
    __slots__ = ("options", "children")
    _non_child = ("dtype", "options")
    children: tuple[Expr, ...]

    def __init__(self, dtype: plc.DataType, options: Any, agg: Expr, *by: Expr) -> None:
        super().__init__(dtype)
        self.options = options
        self.children = (agg, *by)
        raise NotImplementedError("Grouped rolling window not implemented")


class Cast(Expr):
    __slots__ = ("children",)
    _non_child = ("dtype",)
    children: tuple[Expr]

    def __init__(self, dtype: plc.DataType, value: Expr) -> None:
        super().__init__(dtype)
        self.children = (value,)
        if not (
            plc.traits.is_fixed_width(self.dtype)
            and plc.traits.is_fixed_width(value.dtype)
            and plc.unary.is_supported_cast(value.dtype, self.dtype)
        ):
            raise NotImplementedError(
                f"Can't cast {self.dtype.id().name} to {value.dtype.id().name}"
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
        return Column(plc.unary.cast(column.obj, self.dtype)).sorted_like(column)

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        # TODO: Could do with sort-based groupby and segmented filter
        (child,) = self.children
        return child.collect_agg(depth=depth)


class Agg(Expr):
    __slots__ = ("name", "options", "op", "request", "children")
    _non_child = ("dtype", "name", "options")
    children: tuple[Expr, ...]

    def __init__(
        self, dtype: plc.DataType, name: str, options: Any, *children: Expr
    ) -> None:
        super().__init__(dtype)
        self.name = name
        self.options = options
        self.children = children
        if name not in Agg._SUPPORTED:
            raise NotImplementedError(
                f"Unsupported aggregation {name=}"
            )  # pragma: no cover; all valid aggs are supported
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
            raise NotImplementedError(
                f"Unreachable, {name=} is incorrectly listed in _SUPPORTED"
            )  # pragma: no cover
        self.request = req
        op = getattr(self, f"_{name}", None)
        if op is None:
            op = partial(self._reduce, request=req)
        elif name in {"min", "max"}:
            op = partial(op, propagate_nans=options)
        elif name in {"count", "first", "last"}:
            pass
        else:
            raise NotImplementedError(
                f"Unreachable, supported agg {name=} has no implementation"
            )  # pragma: no cover
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
            raise NotImplementedError(
                "Nested aggregations in groupby"
            )  # pragma: no cover; check_agg trips first
        if (isminmax := self.name in {"min", "max"}) and self.options:
            raise NotImplementedError("Nan propagation in groupby for min/max")
        (child,) = self.children
        ((expr, _, _),) = child.collect_agg(depth=depth + 1).requests
        if self.request is None:
            raise NotImplementedError(
                f"Aggregation {self.name} in groupby"
            )  # pragma: no cover; __init__ trips first
        if isminmax and plc.traits.is_floating_point(self.dtype):
            assert expr is not None
            # Ignore nans in these groupby aggs, do this by masking
            # nans in the input
            expr = UnaryFunction(self.dtype, "mask_nans", (), expr)
        return AggInfo([(expr, self.request, self)])

    def _reduce(
        self, column: Column, *, request: plc.aggregation.Aggregation
    ) -> Column:
        return Column(
            plc.Column.from_scalar(
                plc.reduce.reduce(column.obj, request, self.dtype),
                1,
            )
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
            )
        )

    def _min(self, column: Column, *, propagate_nans: bool) -> Column:
        if propagate_nans and column.nan_count > 0:
            return Column(
                plc.Column.from_scalar(
                    plc.interop.from_arrow(
                        pa.scalar(float("nan"), type=plc.interop.to_arrow(self.dtype))
                    ),
                    1,
                )
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
                )
            )
        if column.nan_count > 0:
            column = column.mask_nans()
        return self._reduce(column, request=plc.aggregation.max())

    def _first(self, column: Column) -> Column:
        return Column(plc.copying.slice(column.obj, [0, 1])[0])

    def _last(self, column: Column) -> Column:
        n = column.obj.size()
        return Column(plc.copying.slice(column.obj, [n - 1, n])[0])

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if context is not ExecutionContext.FRAME:
            raise NotImplementedError(
                f"Agg in context {context}"
            )  # pragma: no cover; unreachable
        (child,) = self.children
        return self.op(child.evaluate(df, context=context, mapping=mapping))


class Ternary(Expr):
    __slots__ = ("children",)
    _non_child = ("dtype",)
    children: tuple[Expr, Expr, Expr]

    def __init__(
        self, dtype: plc.DataType, when: Expr, then: Expr, otherwise: Expr
    ) -> None:
        super().__init__(dtype)
        self.children = (when, then, otherwise)

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        when, then, otherwise = (
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        )
        then_obj = then.obj_scalar if then.is_scalar else then.obj
        otherwise_obj = otherwise.obj_scalar if otherwise.is_scalar else otherwise.obj
        return Column(plc.copying.copy_if_else(then_obj, otherwise_obj, when.obj))


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
        if not plc.binaryop.is_supported_operation(
            self.dtype, left.dtype, right.dtype, op
        ):
            raise NotImplementedError(
                f"Operation {op.name} not supported "
                f"for types {left.dtype.id().name} and {right.dtype.id().name} "
                f"with output type {self.dtype.id().name}"
            )

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
