# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""
DSL nodes for the LogicalPlan of polars.

An IR node is either a source, normal, or a sink. Respectively they
can be considered as functions:

- source: `IO () -> DataFrame`
- normal: `DataFrame -> DataFrame`
- sink: `DataFrame -> IO ()`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from typing_extensions import assert_never

import cudf._lib.pylibcudf as plc
import cudf_polars.dsl.expr as expr

if TYPE_CHECKING:
    from cudf_polars.dsl.expr import Expr


__all__ = [
    "IR",
    "PythonScan",
    "Scan",
    "Cache",
    "DataFrameScan",
    "Select",
    "GroupBy",
    "Join",
    "HStack",
    "Distinct",
    "Sort",
    "Slice",
    "Filter",
    "Projection",
    "MapFunction",
    "Union",
    "HConcat",
    "ExtContext",
]


@dataclass(slots=True)
class IR:
    schema: dict


@dataclass(slots=True)
class PythonScan(IR):
    options: Any
    predicate: Expr | None

    def evaluate(self):
        raise NotImplementedError


@dataclass(slots=True)
class Scan(IR):
    typ: Any
    paths: list[str]
    file_options: Any
    predicate: Expr | None

    def __post_init__(self):
        if self.file_options.n_rows is not None:
            raise NotImplementedError("row limit in scan")
        if self.typ not in ("csv", "parquet"):
            raise NotImplementedError(f"Unhandled scan type: {self.typ}")
    def evaluate(self):
        options = self.file_options
        n_rows = options.n_rows
        with_columns = options.with_columns
        row_index = options.row_index
        assert n_rows is None
        if self.typ == "csv":
            df = cudf.concat(
                [cudf.read_csv(p, usecols=with_columns) for p in self.paths]
            )
        elif self.typ == "parquet":
            df = cudf.read_parquet(self.paths, columns=with_columns)
        else:
            assert_never(self.typ)
        if row_index is not None:
            name, offset = row_index
            dtype = self.schema[name]
            index = as_column(
                ..., dtype=dtype
            )
            
        
            
                


@dataclass(slots=True)
class Cache(IR):
    key: int
    value: IR


@dataclass(slots=True)
class DataFrameScan(IR):
    df: Any
    projection: list[str]
    predicate: Expr | None


@dataclass(slots=True)
class Select(IR):
    df: IR
    cse: list[Expr]
    expr: list[Expr]


@dataclass(slots=True)
class GroupBy(IR):
    df: IR
    agg_requests: list[Expr]
    keys: list[Expr]
    options: Any

    @staticmethod
    def check_agg(agg: Expr) -> int:
        """
        Determine if we can handle an aggregation expression.

        Parameters
        ----------
        agg
            Expression to check

        Returns
        -------
        depth of nesting

        Raises
        ------
        NotImplementedError for unsupported expression nodes.
        """
        if isinstance(agg, expr.Agg):
            if agg.name == "implode":
                raise NotImplementedError("implode in groupby")
            return 1 + GroupBy.check_agg(agg.column)
        elif isinstance(agg, (expr.Len, expr.Column, expr.Literal)):
            return 0
        elif isinstance(agg, expr.BinOp):
            return max(GroupBy.check_agg(agg.left), GroupBy.check_agg(agg.right))
        elif isinstance(agg, expr.Cast):
            return GroupBy.check_agg(agg.column)
        else:
            raise NotImplementedError(f"No handler for {agg=}")

    def __post_init__(self):
        """Check whether all the aggregations are implemented."""
        if any(GroupBy.check_agg(a) > 1 for a in self.agg_requests):
            raise NotImplementedError("Nested aggregations in groupby")


@dataclass(slots=True)
class Join(IR):
    left: IR
    right: IR
    left_on: list[Expr]
    right_on: list[Expr]
    options: Any

    def __post_init__(self):
        """Raise for unsupported options."""
        how, coalesce = self.options[0], self.options[-1]
        if how == "cross":
            raise NotImplementedError("cross join not implemented")
        if how == "outer" and not coalesce:
            raise NotImplementedError("non-coalescing outer join")


@dataclass(slots=True)
class HStack(IR):
    df: IR
    columns: list[Expr]


@dataclass(slots=True)
class Distinct(IR):
    df: IR
    options: Any


@dataclass(slots=True)
class Sort(IR):
    df: IR
    by: list[Expr]
    options: Any


@dataclass(slots=True)
class Slice(IR):
    df: IR
    offset: int
    length: int


@dataclass(slots=True)
class Filter(IR):
    df: IR
    mask: Expr


@dataclass(slots=True)
class Projection(IR):
    df: IR


@dataclass(slots=True)
class MapFunction(IR):
    df: IR
    name: str
    options: Any


@dataclass(slots=True)
class Union(IR):
    dfs: list[IR]


@dataclass(slots=True)
class HConcat(IR):
    dfs: list[IR]


@dataclass(slots=True)
class ExtContext(IR):
    df: IR
    extra: list[IR]
