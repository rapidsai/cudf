# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Rolling DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.expressions.base import Expr

if TYPE_CHECKING:
    import pylibcudf as plc

    from cudf_polars.dsl.expressions.base import ExecutionContext

__all__ = ["GroupedRollingWindow", "RollingWindow"]


class RollingWindow(Expr):
    __slots__ = ("options",)
    _non_child = ("dtype", "context", "options")

    def __init__(
        self, dtype: plc.DataType, context: ExecutionContext, options: Any, agg: Expr
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.context = context
        self.children = (agg,)
        self.is_pointwise = False
        raise NotImplementedError("Rolling window not implemented")


class GroupedRollingWindow(Expr):
    __slots__ = ("options",)
    _non_child = ("dtype", "context", "options")

    def __init__(
        self,
        dtype: plc.DataType,
        context: ExecutionContext,
        options: Any,
        agg: Expr,
        *by: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.context = context
        self.children = (agg, *by)
        self.is_pointwise = False
        raise NotImplementedError("Grouped rolling window not implemented")
