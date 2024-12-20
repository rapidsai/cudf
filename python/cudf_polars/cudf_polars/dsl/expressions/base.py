# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Base and common classes for expression DSL nodes."""

from __future__ import annotations

import enum
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.nodebase import Node

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.containers import Column, DataFrame

__all__ = ["Expr", "NamedExpr", "Col", "AggInfo", "ExecutionContext", "ColRef"]


class AggInfo(NamedTuple):
    requests: list[tuple[Expr | None, plc.aggregation.Aggregation, Expr]]


class ExecutionContext(IntEnum):
    FRAME = enum.auto()
    GROUPBY = enum.auto()
    ROLLING = enum.auto()


class Expr(Node["Expr"]):
    """An abstract expression object."""

    __slots__ = ("dtype",)
    dtype: plc.DataType
    """Data type of the expression."""
    # This annotation is needed because of https://github.com/python/mypy/issues/17981
    _non_child: ClassVar[tuple[str, ...]] = ("dtype",)
    """Names of non-child data (not Exprs) for reconstruction."""

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


class ErrorExpr(Expr):
    __slots__ = ("error",)
    _non_child = ("dtype", "error")
    error: str

    def __init__(self, dtype: plc.DataType, error: str) -> None:
        self.dtype = dtype
        self.error = error
        self.children = ()


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
    ) -> Column:
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
        Evaluated Column with name attached.

        See Also
        --------
        :meth:`Expr.evaluate` for details, this function just adds the
        name to a column produced from an expression.
        """
        return self.value.evaluate(df, context=context, mapping=mapping).rename(
            self.name
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return self.value.collect_agg(depth=depth)


class Col(Expr):
    __slots__ = ("name",)
    _non_child = ("dtype", "name")
    name: str

    def __init__(self, dtype: plc.DataType, name: str) -> None:
        self.dtype = dtype
        self.name = name
        self.children = ()

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        # Deliberately remove the name here so that we guarantee
        # evaluation of the IR produces names.
        return df.column_map[self.name].rename(None)

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return AggInfo([(self, plc.aggregation.collect_list(), self)])


class ColRef(Expr):
    __slots__ = ("index", "table_ref")
    _non_child = ("dtype", "index", "table_ref")
    index: int
    table_ref: plc.expressions.TableReference

    def __init__(
        self,
        dtype: plc.DataType,
        index: int,
        table_ref: plc.expressions.TableReference,
        column: Expr,
    ) -> None:
        if not isinstance(column, Col):
            raise TypeError("Column reference should only apply to columns")
        self.dtype = dtype
        self.index = index
        self.table_ref = table_ref
        self.children = (column,)

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        raise NotImplementedError(
            "Only expect this node as part of an expression translated to libcudf AST."
        )
