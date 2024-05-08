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

from dataclasses import dataclass
from typing import Any

__all__ = [
    "Expr",
    "NamedExpr",
    "Literal",
    "Column",
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


@dataclass(slots=True)
class Expr:
    pass


@dataclass(slots=True)
class NamedExpr(Expr):
    name: str
    value: Expr


@dataclass(slots=True)
class Literal(Expr):
    dtype: Any
    value: Any


@dataclass(slots=True)
class Column(Expr):
    name: str


@dataclass(slots=True)
class Len(Expr):
    pass


@dataclass(slots=True)
class BooleanFunction(Expr):
    name: str
    options: Any
    arguments: list[Expr]


@dataclass(slots=True)
class Sort(Expr):
    column: Expr
    options: Any


@dataclass(slots=True)
class SortBy(Expr):
    column: Expr
    by: list[Expr]
    descending: list[bool]


@dataclass(slots=True)
class Gather(Expr):
    values: Expr
    indices: Expr


@dataclass(slots=True)
class Filter(Expr):
    values: Expr
    mask: Expr


@dataclass(slots=True)
class Window(Expr):
    agg: Expr
    by: None | list[Expr]
    options: Any


@dataclass(slots=True)
class Cast(Expr):
    dtype: Any
    column: Expr


@dataclass(slots=True)
class Agg(Expr):
    column: Expr
    name: str
    options: Any


@dataclass(slots=True)
class BinOp(Expr):
    left: Expr
    right: Expr
    op: Any
