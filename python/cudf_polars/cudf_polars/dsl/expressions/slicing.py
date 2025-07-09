# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Slicing DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.expressions.base import (
    ExecutionContext,
    Expr,
)

if TYPE_CHECKING:
    from cudf_polars.containers import Column, DataFrame, DataType


__all__ = ["Slice"]


class Slice(Expr):
    __slots__ = ("length", "offset")
    _non_child = ("dtype", "offset", "length")

    def __init__(
        self,
        dtype: DataType,
        offset: int,
        length: int | None,
        column: Expr,
    ) -> None:
        self.dtype = dtype
        self.offset = offset
        self.length = length
        self.children = (column,)
        self.is_pointwise = False

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context)
        return column.slice((self.offset, self.length))
