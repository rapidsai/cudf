# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Rolling DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.utils.windows import range_window_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

    from cudf_polars.typing import ClosedInterval

__all__ = ["GroupedRollingWindow", "RollingWindow"]


class RollingWindow(Expr):
    __slots__ = (
        "agg_names",
        "closed_window",
        "following",
        "index",
        "preceding",
    )
    _non_child = (
        "dtype",
        "index",
        "preceding",
        "following",
        "closed_window",
        "agg_names",
    )
    closed_window: ClosedInterval

    def __init__(
        self,
        dtype: plc.DataType,
        preceding: pa.Scalar,
        following: pa.Scalar,
        closed_window: ClosedInterval,
        agg_names: Sequence[str],
        orderby: Expr,
        post_agg: Expr,
        *aggs: Expr,
    ) -> None:
        self.dtype = dtype
        self.preceding = preceding
        self.following = following
        self.closed_window = closed_window
        self.children = (orderby, post_agg, *aggs)
        self.is_pointwise = False
        self.agg_names = tuple(agg_names)

    def do_evaluate(  # noqa: D102
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        if context != ExecutionContext.FRAME:
            raise NotImplementedError(f"Cannot evaluate rolling in context {context}")
        orderby_expr, post_agg, *aggs = self.children
        orderby = orderby_expr.evaluate(df, context=context)
        requests = []
        for agg in aggs:
            if isinstance(agg, expr.Len):
                col = orderby.obj
            elif isinstance(agg, expr.Agg):
                (child,) = agg.children
                col = child.evaluate(df, context=ExecutionContext.ROLLING).obj
            else:
                col = agg.evaluate(df, context=ExecutionContext.ROLLING).obj
            requests.append(plc.rolling.RollingRequest(col, agg.agg_request))
        preceding, following = range_window_bounds(
            self.preceding, self.following, self.closed_window
        )
        values = plc.rolling.grouped_range_rolling_window(
            plc.Table([]),
            orderby.obj,
            orderby.order,
            plc.types.NullOrder.AFTER,
            preceding,
            following,
            1,
            requests,
        )
        if isinstance(post_agg, expr.Col):
            (result,) = values.columns()
            return Column(result)
        agg_df = DataFrame(
            Column(col, name=name)
            for col, name in zip(values.columns(), self.agg_names, strict=True)
        )
        return post_agg.evaluate(agg_df, context=ExecutionContext.ROLLING)


class GroupedRollingWindow(Expr):
    __slots__ = ("options",)
    _non_child = ("dtype", "options")

    def __init__(self, dtype: plc.DataType, options: Any, agg: Expr, *by: Expr) -> None:
        self.dtype = dtype
        self.options = options
        self.children = (agg, *by)
        self.is_pointwise = False
        raise NotImplementedError("Grouped rolling window not implemented")
