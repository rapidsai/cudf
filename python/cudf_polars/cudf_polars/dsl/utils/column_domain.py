# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column value domains between IR nodes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    Cache,
    Distinct,
    Filter,
    GroupBy,
    HStack,
    Join,
    Projection,
    Select,
    Slice,
    Sort,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.dsl.ir import IR

__all__ = ["ColumnRef", "column_domain_bindings"]


@dataclass(frozen=True)
class ColumnRef:
    """A named column produced by an IR node."""

    node: IR
    name: str


@singledispatch
def column_domain_bindings(node: IR) -> Mapping[str, ColumnRef]:
    """
    Map output columns to child columns containing their value domains.

    For every ``output_name -> ColumnRef(child, input_name)`` binding, every
    value appearing in ``node[output_name]`` is guaranteed to appear in
    ``child[input_name]``. Row order, multiplicity, and cardinality are not
    preserved.

    If a name in ``node.schema`` does not appear in the mapping it means
    that it was not possible to derive a relationship between the domain of
    the output and input values for that column.
    """
    return {}


@column_domain_bindings.register(Select)
def _(node: Select) -> Mapping[str, ColumnRef]:
    child = node.children[0]
    return {
        item.name: ColumnRef(child, item.value.name)
        for item in node.exprs
        if isinstance(item.value, expr.Col)
    }


@column_domain_bindings.register(HStack)
def _(node: HStack) -> Mapping[str, ColumnRef]:
    child = node.children[0]
    replaced = {item.name for item in node.columns}
    return {
        name: ColumnRef(child, name) for name in child.schema if name not in replaced
    } | {
        item.name: ColumnRef(child, item.value.name)
        for item in node.columns
        if isinstance(item.value, expr.Col)
    }


@column_domain_bindings.register(GroupBy)
def _(node: GroupBy) -> Mapping[str, ColumnRef]:
    child = node.children[0]
    return {
        key.name: ColumnRef(child, key.value.name)
        for key in node.keys
        if isinstance(key.value, expr.Col)
    }


@column_domain_bindings.register(Join)
def _(node: Join) -> Mapping[str, ColumnRef]:
    left, right = node.children
    how = node.options[0]
    if how in ("Semi", "Anti"):
        return {
            name: ColumnRef(left, name) for name in node.schema if name in left.schema
        }
    if how != "Inner":
        return {}

    bindings = {name: ColumnRef(left, name) for name in left.schema}
    suffix = node.options[3]
    for name in right.schema:
        output_name = f"{name}{suffix}" if name in left.schema else name
        if output_name in node.schema:
            bindings[output_name] = ColumnRef(right, name)
    return bindings


@column_domain_bindings.register(Cache)
@column_domain_bindings.register(Distinct)
@column_domain_bindings.register(Filter)
@column_domain_bindings.register(Projection)
@column_domain_bindings.register(Slice)
@column_domain_bindings.register(Sort)
def _(
    node: Cache | Distinct | Filter | Projection | Slice | Sort,
) -> Mapping[str, ColumnRef]:
    child = node.children[0]
    return {
        name: ColumnRef(child, name) for name in node.schema if name in child.schema
    }
