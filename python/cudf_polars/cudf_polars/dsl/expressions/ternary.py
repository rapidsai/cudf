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
        then_obj = then.obj_scalar(stream=df.stream) if then.is_scalar else then.obj
        otherwise_obj = (
            otherwise.obj_scalar(stream=df.stream)
            if otherwise.is_scalar
            else otherwise.obj
        )
        return Column(
            plc.copying.copy_if_else(
                then_obj, otherwise_obj, when.obj, stream=df.stream
            ),
            dtype=self.dtype,
        )
