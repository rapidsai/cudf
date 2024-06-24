# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Typing utilities for cudf_polars."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

from polars.polars import _expr_nodes as pl_expr, _ir_nodes as pl_ir

import cudf._lib.pylibcudf as plc

if TYPE_CHECKING:
    from typing import Callable

    import polars as pl

IR: TypeAlias = (
    pl_ir.PythonScan
    | pl_ir.Scan
    | pl_ir.Cache
    | pl_ir.DataFrameScan
    | pl_ir.Select
    | pl_ir.GroupBy
    | pl_ir.Join
    | pl_ir.HStack
    | pl_ir.Distinct
    | pl_ir.Sort
    | pl_ir.Slice
    | pl_ir.Filter
    | pl_ir.SimpleProjection
    | pl_ir.MapFunction
    | pl_ir.Union
    | pl_ir.HConcat
    | pl_ir.ExtContext
)

Expr: TypeAlias = (
    pl_expr.Function
    | pl_expr.Window
    | pl_expr.Literal
    | pl_expr.Sort
    | pl_expr.SortBy
    | pl_expr.Gather
    | pl_expr.Filter
    | pl_expr.Cast
    | pl_expr.Column
    | pl_expr.Agg
    | pl_expr.BinaryExpr
    | pl_expr.Len
    | pl_expr.PyExprIR
)

Schema: TypeAlias = Mapping[str, plc.DataType]


class NodeTraverser(Protocol):
    """Abstract protocol for polars NodeTraverser."""

    def get_node(self) -> int:
        """Return current plan node id."""
        ...

    def set_node(self, n: int) -> None:
        """Set the current plan node to n."""
        ...

    def view_current_node(self) -> IR:
        """Convert current plan node to python rep."""
        ...

    def get_schema(self) -> Mapping[str, pl.DataType]:
        """Get the schema of the current plan node."""
        ...

    def get_dtype(self, n: int) -> pl.DataType:
        """Get the datatype of the given expression id."""
        ...

    def view_expression(self, n: int) -> Expr:
        """Convert the given expression to python rep."""
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
