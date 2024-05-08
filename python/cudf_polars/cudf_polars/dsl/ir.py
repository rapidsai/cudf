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


@dataclass(slots=True)
class Scan(IR):
    typ: Any
    paths: list[str]
    file_options: Any
    predicate: Expr | None


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


@dataclass(slots=True)
class Join(IR):
    left: IR
    right: IR
    left_on: list[Expr]
    right_on: list[Expr]
    options: Any


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
