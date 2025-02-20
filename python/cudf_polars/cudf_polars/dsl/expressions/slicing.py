# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Slicing DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc
import pyarrow as pa

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import (
    ExecutionContext,
    Expr,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.containers import DataFrame


__all__ = ["Slice"]


class Slice(Expr):
    __slots__ = ("offset", "length")
    _non_child = ("dtype", "offset", "length")

    def __init__(
        self, dtype: plc.DataType, offset: int, length: int, column: Expr,
    ) -> None:
        self.dtype = dtype
        self.offset = offset
        self.length = length

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        return df.slice((self.offset, self.length)).columns[0]
