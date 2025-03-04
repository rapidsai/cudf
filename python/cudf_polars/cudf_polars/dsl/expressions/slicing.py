# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Slicing DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.expressions.base import Expr

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pylibcudf as plc

    from cudf_polars.containers import Column, DataFrame
    from cudf_polars.dsl.expressions.base import ExecutionContext


__all__ = ["Slice"]


class Slice(Expr):
    __slots__ = ("length", "offset")
    _non_child = ("dtype", "context", "offset", "length")

    def __init__(
        self,
        dtype: plc.DataType,
        context: ExecutionContext,
        offset: int,
        length: int,
        column: Expr,
    ) -> None:
        self.dtype = dtype
        self.context = context
        self.offset = offset
        self.length = length
        self.children = (column,)

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context, mapping=mapping)
        return column.slice((self.offset, self.length))
