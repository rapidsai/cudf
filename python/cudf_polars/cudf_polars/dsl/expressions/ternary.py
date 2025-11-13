# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for ternary operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import (
    ExecutionContext,
    Expr,
)
from cudf_polars.dsl.utils.reshape import broadcast

if TYPE_CHECKING:
    from cudf_polars.containers import DataFrame, DataType


__all__ = ["Ternary"]


class Ternary(Expr):
    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(
        self, dtype: DataType, when: Expr, then: Expr, otherwise: Expr
    ) -> None:
        self.dtype = dtype
        self.children = (when, then, otherwise)
        self.is_pointwise = True

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        when, then, otherwise = (
            child.evaluate(df, context=context) for child in self.children
        )
        if when.is_scalar:
            # For scalar predicates: lowering to copy_if_else would require
            # materializing an all true/false mask column. Instead, just pick
            # the correct branch.
            when_predicate = when.obj_scalar(stream=df.stream).to_py()
            pick, other = (then, otherwise) if when_predicate else (otherwise, then)

            if pick.is_scalar:
                (pick_col,) = broadcast(
                    pick,
                    target_length=1 if other.is_scalar else other.size,
                    stream=df.stream,
                )
                return Column(pick_col.obj, dtype=self.dtype)

            return Column(pick.obj, dtype=self.dtype)

        then_col, otherwise_col = broadcast(
            then, otherwise, target_length=when.size, stream=df.stream
        )

        return Column(
            plc.copying.copy_if_else(
                then_col.obj,
                otherwise_col.obj,
                when.obj,
                stream=df.stream,
            ),
            dtype=self.dtype,
        )
