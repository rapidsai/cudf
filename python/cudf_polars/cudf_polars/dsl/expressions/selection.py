# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for selection operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import Expr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expressions.base import ExecutionContext

__all__ = ["Filter", "Gather"]


class Gather(Expr):
    __slots__ = ()
    _non_child = ("dtype", "context")

    def __init__(
        self,
        dtype: plc.DataType,
        context: ExecutionContext,
        values: Expr,
        indices: Expr,
    ) -> None:
        self.dtype = dtype
        self.context = context
        self.children = (values, indices)
        self.is_pointwise = False

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        values, indices = (
            child.evaluate(df, mapping=mapping) for child in self.children
        )
        lo, hi = plc.reduce.minmax(indices.obj)
        lo = plc.interop.to_arrow(lo).as_py()
        hi = plc.interop.to_arrow(hi).as_py()
        n = df.num_rows
        if hi >= n or lo < -n:
            raise ValueError("gather indices are out of bounds")
        if indices.null_count:
            bounds_policy = plc.copying.OutOfBoundsPolicy.NULLIFY
            obj = plc.replace.replace_nulls(
                indices.obj,
                plc.interop.from_arrow(
                    pa.scalar(n, type=plc.interop.to_arrow(indices.obj.type()))
                ),
            )
        else:
            bounds_policy = plc.copying.OutOfBoundsPolicy.DONT_CHECK
            obj = indices.obj
        table = plc.copying.gather(plc.Table([values.obj]), obj, bounds_policy)
        return Column(table.columns()[0])


class Filter(Expr):
    __slots__ = ()
    _non_child = ("dtype", "context")

    def __init__(
        self,
        dtype: plc.DataType,
        context: ExecutionContext,
        values: Expr,
        indices: Expr,
    ):
        self.dtype = dtype
        self.children = (values, indices)
        self.context = context
        self.is_pointwise = True

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        values, mask = (child.evaluate(df, mapping=mapping) for child in self.children)
        table = plc.stream_compaction.apply_boolean_mask(
            plc.Table([values.obj]), mask.obj
        )
        return Column(table.columns()[0]).sorted_like(values)
