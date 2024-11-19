# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Literal DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import AggInfo, ExecutionContext, Expr
from cudf_polars.utils import dtypes

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import pyarrow as pa

    import polars as pl

    from cudf_polars.containers import DataFrame

__all__ = ["Literal", "LiteralColumn"]


class Literal(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: pa.Scalar[Any]

    def __init__(self, dtype: plc.DataType, value: pa.Scalar[Any]) -> None:
        self.dtype = dtype
        assert value.type == plc.interop.to_arrow(dtype)
        self.value = value
        self.children = ()

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        # datatype of pyarrow scalar is correct by construction.
        return Column(plc.Column.from_scalar(plc.interop.from_arrow(self.value), 1))

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return AggInfo([])


class LiteralColumn(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: pa.Array[Any]

    def __init__(self, dtype: plc.DataType, value: pl.Series) -> None:
        self.dtype = dtype
        data = value.to_arrow()
        self.value = data.cast(dtypes.downcast_arrow_lists(data.type))
        self.children = ()

    def get_hashable(self) -> Hashable:
        """Compute a hash of the column."""
        # This is stricter than necessary, but we only need this hash
        # for identity in groupby replacements so it's OK. And this
        # way we avoid doing potentially expensive compute.
        return (type(self), self.dtype, id(self.value))

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        # datatype of pyarrow array is correct by construction.
        return Column(plc.interop.from_arrow(self.value))

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        return AggInfo([])
