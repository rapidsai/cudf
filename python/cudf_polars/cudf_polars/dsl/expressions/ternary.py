# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
    from collections.abc import Mapping

    from cudf_polars.containers import DataFrame


__all__ = ["Ternary"]


class Ternary(Expr):
    __slots__ = ("children",)
    _non_child = ("dtype",)
    children: tuple[Expr, Expr, Expr]

    def __init__(
        self, dtype: plc.DataType, when: Expr, then: Expr, otherwise: Expr
    ) -> None:
        super().__init__(dtype)
        self.children = (when, then, otherwise)

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        when, then, otherwise = (
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        )
        then_obj = then.obj_scalar if then.is_scalar else then.obj
        otherwise_obj = otherwise.obj_scalar if otherwise.is_scalar else otherwise.obj
        return Column(plc.copying.copy_if_else(then_obj, otherwise_obj, when.obj))
