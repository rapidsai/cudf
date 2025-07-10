# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Literal DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from collections.abc import Hashable

    from cudf_polars.containers import DataFrame

__all__ = ["Literal", "LiteralColumn"]


class Literal(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: Any  # Python scalar

    def __init__(self, dtype: DataType, value: Any) -> None:
        if value is None and dtype.id() == plc.TypeId.EMPTY:
            # TypeId.EMPTY not supported by libcudf
            # cuDF Python also maps EMPTY to INT8
            dtype = DataType(pl.datatypes.Int8())
        self.dtype = dtype
        self.value = value
        self.children = ()
        self.is_pointwise = True

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return Column(
            plc.Column.from_scalar(plc.Scalar.from_py(self.value, self.dtype.plc), 1),
            dtype=self.dtype,
        )

    @property
    def agg_request(self) -> NoReturn:  # noqa: D102
        raise NotImplementedError(
            "Not expecting to require agg request of literal"
        )  # pragma: no cover

    def astype(self, dtype: DataType) -> Literal:
        """Cast self to dtype."""
        if self.value is None:
            return Literal(dtype, self.value)
        else:
            # Use polars to cast instead of pylibcudf
            # since there are just Python scalars
            casted = pl.Series(values=[self.value], dtype=self.dtype.polars).cast(
                dtype.polars
            )[0]
            return Literal(dtype, casted)


class LiteralColumn(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: pl.Series

    def __init__(self, dtype: DataType, value: pl.Series) -> None:
        self.dtype = dtype
        self.value = value
        self.children = ()
        self.is_pointwise = True

    def get_hashable(self) -> Hashable:
        """Compute a hash of the column."""
        # This is stricter than necessary, but we only need this hash
        # for identity in groupby replacements so it's OK. And this
        # way we avoid doing potentially expensive compute.
        return (type(self), self.dtype.plc, id(self.value))

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return Column(plc.Column.from_arrow(self.value), dtype=self.dtype)

    @property
    def agg_request(self) -> NoReturn:  # noqa: D102
        raise NotImplementedError(
            "Not expecting to require agg request of literal"
        )  # pragma: no cover
