# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Typing utilities for cudf_polars."""

from __future__ import annotations

import sys
from collections.abc import Hashable, MutableMapping
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NewType,
    Protocol,
    TypeVar,
    Union,
)

import polars as pl
from polars.polars import _expr_nodes as pl_expr, _ir_nodes as pl_ir

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping
    from typing import TypeAlias

    import pylibcudf as plc

    from cudf_polars.containers import DataFrame, DataType
    from cudf_polars.dsl import expr, ir, nodebase
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.utils.config import ConfigOptions


if sys.version_info >= (3, 11):
    # Inheriting from TypeDict + Generic added in python 3.11
    from typing import TypedDict  # pragma: no cover
else:
    from typing_extensions import TypedDict  # pragma: no cover


__all__: list[str] = [
    "ClosedInterval",
    "ColumnHeader",
    "ColumnOptions",
    "DataFrameHeader",
    "ExprTransformer",
    "GenericTransformer",
    "IRTransformer",
    "NodeTraverser",
    "OptimizationArgs",
    "PolarsExpr",
    "PolarsIR",
    "Schema",
    "Slice",
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

PolarsSchema: TypeAlias = dict[str, pl.DataType]
Schema: TypeAlias = dict[str, "DataType"]

Slice: TypeAlias = tuple[int, int | None]

CSECache: TypeAlias = MutableMapping[int, tuple["DataFrame", int]]

ClosedInterval: TypeAlias = Literal["left", "right", "both", "none"]

Duration = NewType("Duration", tuple[int, int, int, int, bool, bool])


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

    def get_schema(self) -> PolarsSchema:
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
StateT_co = TypeVar(
    "StateT_co",
    bound="CachingVisitorState",
    covariant=True,
)


class GenericTransformer(Protocol[U_contra, V_co, StateT_co]):
    """Abstract protocol for recursive visitors."""

    def __call__(self, __value: U_contra) -> V_co:
        """Apply the visitor to the node."""
        ...

    @property
    def state(self) -> StateT_co:
        """Arbitrary immutable state."""
        ...


class ASTState(TypedDict):
    """
    State for AST transformation in :mod:`cudf_polars.dsl.to_ast`.

    Parameters
    ----------
    for_parquet
        Indicator for whether this transformation should provide an expression
        suitable for use in parquet filters.

        If ``for_parquet`` is ``False``, the dictionary should contain a
    """

    for_parquet: bool


class GenericState(Generic[NodeT], TypedDict):
    replacements: Mapping[NodeT, NodeT]


class ExprExprState(TypedDict):
    """
    State used for AST transformation in :mod:`cudf_polars.dsl.to_ast`.

    Parameters
    ----------
    name_to_index
        Mapping from column names to column indices in the table
        eventually used for evaluation.

    table_ref
        pylibcudf `TableReference` indicating whether column
        references are coming from the left or right table.
    """

    name_to_index: Mapping[str, int]
    table_ref: plc.expressions.TableReference


class ExprDecomposerState(TypedDict):
    """State for ExprDecomposer."""

    input_ir: ir.IR
    input_partition_info: PartitionInfo
    config_options: ConfigOptions
    unique_names: Generator[str, None, None]


class LowerIRState(TypedDict):
    config_options: ConfigOptions


CachingVisitorState: TypeAlias = (
    ExprExprState | ExprDecomposerState | LowerIRState | GenericState | ASTState
)


# Quotes to avoid circular import
ExprTransformer: TypeAlias = GenericTransformer["expr.Expr", "expr.Expr", ExprExprState]
"""Protocol for transformation of Expr nodes."""

IRTransformer: TypeAlias = GenericTransformer["ir.IR", "ir.IR", CachingVisitorState]
"""Protocol for transformation of IR nodes."""

LowerIRTransformer: TypeAlias = GenericTransformer[
    "ir.IR", "tuple[ir.IR, MutableMapping[ir.IR, PartitionInfo]]", LowerIRState
]
"""Protocol for Lowering IR nodes."""

ExprDecomposer: TypeAlias = "GenericTransformer[expr.Expr, tuple[expr.Expr, ir.IR, MutableMapping[ir.IR, PartitionInfo]], ExprDecomposerState]"
"""Protocol for decomposing expressions."""


class ColumnOptions(TypedDict):
    """
    Column constructor options.

    Notes
    -----
    Used to serialize Column and DataFrame containers.
    """

    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder
    name: str | None
    dtype: str | None


class DeserializedColumnOptions(TypedDict):
    """
    Deserialized Column constructor options.

    Notes
    -----
    Used to deserialize Column and DataFrame containers.
    """

    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder
    name: str | None
    dtype: DataType | None


class ColumnHeader(TypedDict):
    """Column serialization header."""

    column_kwargs: ColumnOptions
    frame_count: int


class DataFrameHeader(TypedDict):
    """DataFrame serialization header."""

    columns_kwargs: list[ColumnOptions]
    frame_count: int
