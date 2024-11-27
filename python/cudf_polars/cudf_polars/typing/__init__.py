# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Typing utilities for cudf_polars."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, Union

from polars.polars import _expr_nodes as pl_expr, _ir_nodes as pl_ir

import pylibcudf as plc

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    import polars as pl

    from cudf_polars.dsl import expr, ir, nodebase

__all__: list[str] = [
    "ExprTransformer",
    "GenericTransformer",
    "IRTransformer",
    "NodeTraverser",
    "OptimizationArgs",
    "PolarsExpr",
    "PolarsIR",
]

PolarsIR: TypeAlias = Union[
    pl_ir.PythonScan,
    pl_ir.Scan,
    pl_ir.Cache,
    pl_ir.DataFrameScan,
    pl_ir.Select,
    pl_ir.GroupBy,
    pl_ir.Join,
    pl_ir.HStack,
    pl_ir.Distinct,
    pl_ir.Sort,
    pl_ir.Slice,
    pl_ir.Filter,
    pl_ir.SimpleProjection,
    pl_ir.MapFunction,
    pl_ir.Union,
    pl_ir.HConcat,
    pl_ir.ExtContext,
]

PolarsExpr: TypeAlias = Union[
    pl_expr.Function,
    pl_expr.Window,
    pl_expr.Literal,
    pl_expr.Sort,
    pl_expr.SortBy,
    pl_expr.Gather,
    pl_expr.Filter,
    pl_expr.Cast,
    pl_expr.Column,
    pl_expr.Agg,
    pl_expr.BinaryExpr,
    pl_expr.Len,
    pl_expr.PyExprIR,
]

Schema: TypeAlias = Mapping[str, plc.DataType]


class NodeTraverser(Protocol):
    """Abstract protocol for polars NodeTraverser."""

    def get_node(self) -> int:
        """Return current plan node id."""
        ...

    def set_node(self, n: int) -> None:
        """Set the current plan node to n."""
        ...

    def view_current_node(self) -> PolarsIR:
        """Convert current plan node to python rep."""
        ...

    def get_schema(self) -> Mapping[str, pl.DataType]:
        """Get the schema of the current plan node."""
        ...

    def get_dtype(self, n: int) -> pl.DataType:
        """Get the datatype of the given expression id."""
        ...

    def view_expression(self, n: int) -> PolarsExpr:
        """Convert the given expression to python rep."""
        ...

    def version(self) -> tuple[int, int]:
        """The IR version as `(major, minor)`."""
        ...

    def set_udf(
        self,
        callback: Callable[[list[str] | None, str | None, int | None], pl.DataFrame],
    ) -> None:
        """Set the callback replacing the current node in the plan."""
        ...


OptimizationArgs: TypeAlias = Literal[
    "type_coercion",
    "predicate_pushdown",
    "projection_pushdown",
    "simplify_expression",
    "slice_pushdown",
    "comm_subplan_elim",
    "comm_subexpr_elim",
    "cluster_with_columns",
    "no_optimization",
]


U_contra = TypeVar("U_contra", bound=Hashable, contravariant=True)
V_co = TypeVar("V_co", covariant=True)
NodeT = TypeVar("NodeT", bound="nodebase.Node[Any]")


class GenericTransformer(Protocol[U_contra, V_co]):
    """Abstract protocol for recursive visitors."""

    def __call__(self, __value: U_contra) -> V_co:
        """Apply the visitor to the node."""
        ...

    @property
    def state(self) -> Mapping[str, Any]:
        """Arbitrary immutable state."""
        ...


# Quotes to avoid circular import
ExprTransformer: TypeAlias = GenericTransformer["expr.Expr", "expr.Expr"]
"""Protocol for transformation of Expr nodes."""

IRTransformer: TypeAlias = GenericTransformer["ir.IR", "ir.IR"]
"""Protocol for transformation of IR nodes."""
