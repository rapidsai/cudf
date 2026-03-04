# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Literal DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from collections.abc import Hashable

    import pyarrow as pa

    from cudf_polars.containers import DataFrame

__all__ = ["Literal", "LiteralColumn"]


class Literal(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: Any  # Python scalar

    def __init__(self, dtype: plc.DataType, value: Any) -> None:
        if value is None and dtype.id() == plc.TypeId.EMPTY:
            # TypeId.EMPTY not supported by libcudf
            # cuDF Python also maps EMPTY to INT8
            dtype = plc.DataType(plc.TypeId.INT8)
        self.dtype = dtype
        self.value = value
        self.children = ()
        self.is_pointwise = True

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return Column(
            plc.Column.from_scalar(plc.Scalar.from_py(self.value, self.dtype), 1)
        )

    @property
    def agg_request(self) -> NoReturn:  # noqa: D102
        raise NotImplementedError(
            "Not expecting to require agg request of literal"
        )  # pragma: no cover


class LiteralColumn(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: pa.Array[Any]

    def __init__(self, dtype: plc.DataType, value: pa.Array) -> None:
        self.dtype = dtype
        self.value = value
        self.children = ()
        self.is_pointwise = True

    def get_hashable(self) -> Hashable:
        """Compute a hash of the column."""
        # This is stricter than necessary, but we only need this hash
        # for identity in groupby replacements so it's OK. And this
        # way we avoid doing potentially expensive compute.
        return (type(self), self.dtype, id(self.value))

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        # datatype of pyarrow array is correct by construction.
        return Column(plc.interop.from_arrow(self.value))

    @property
    def agg_request(self) -> NoReturn:  # noqa: D102
        raise NotImplementedError(
            "Not expecting to require agg request of literal"
        )  # pragma: no cover
