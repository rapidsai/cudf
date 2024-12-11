# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for datetime operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from polars.polars import _expr_nodes as pl_expr

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.containers import DataFrame

__all__ = ["TemporalFunction"]


class TemporalFunction(Expr):
    __slots__ = ("name", "options")
    _COMPONENT_MAP: ClassVar[
        dict[pl_expr.TemporalFunction, plc.datetime.DatetimeComponent]
    ] = {
        pl_expr.TemporalFunction.Year: plc.datetime.DatetimeComponent.YEAR,
        pl_expr.TemporalFunction.Month: plc.datetime.DatetimeComponent.MONTH,
        pl_expr.TemporalFunction.Day: plc.datetime.DatetimeComponent.DAY,
        pl_expr.TemporalFunction.WeekDay: plc.datetime.DatetimeComponent.WEEKDAY,
        pl_expr.TemporalFunction.Hour: plc.datetime.DatetimeComponent.HOUR,
        pl_expr.TemporalFunction.Minute: plc.datetime.DatetimeComponent.MINUTE,
        pl_expr.TemporalFunction.Second: plc.datetime.DatetimeComponent.SECOND,
        pl_expr.TemporalFunction.Millisecond: plc.datetime.DatetimeComponent.MILLISECOND,
        pl_expr.TemporalFunction.Microsecond: plc.datetime.DatetimeComponent.MICROSECOND,
        pl_expr.TemporalFunction.Nanosecond: plc.datetime.DatetimeComponent.NANOSECOND,
    }
    _non_child = ("dtype", "name", "options")

    def __init__(
        self,
        dtype: plc.DataType,
        name: pl_expr.TemporalFunction,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        if self.name not in self._COMPONENT_MAP:
            raise NotImplementedError(f"Temporal function {self.name}")

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = [
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        ]
        (column,) = columns
        if self.name == pl_expr.TemporalFunction.Microsecond:
            millis = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MILLISECOND
            )
            micros = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MICROSECOND
            )
            millis_as_micros = plc.binaryop.binary_operation(
                millis,
                plc.interop.from_arrow(pa.scalar(1_000, type=pa.int32())),
                plc.binaryop.BinaryOperator.MUL,
                plc.DataType(plc.TypeId.INT32),
            )
            total_micros = plc.binaryop.binary_operation(
                micros,
                millis_as_micros,
                plc.binaryop.BinaryOperator.ADD,
                plc.types.DataType(plc.types.TypeId.INT32),
            )
            return Column(total_micros)
        elif self.name == pl_expr.TemporalFunction.Nanosecond:
            millis = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MILLISECOND
            )
            micros = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MICROSECOND
            )
            nanos = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.NANOSECOND
            )
            millis_as_nanos = plc.binaryop.binary_operation(
                millis,
                plc.interop.from_arrow(pa.scalar(1_000_000, type=pa.int32())),
                plc.binaryop.BinaryOperator.MUL,
                plc.types.DataType(plc.types.TypeId.INT32),
            )
            micros_as_nanos = plc.binaryop.binary_operation(
                micros,
                plc.interop.from_arrow(pa.scalar(1_000, type=pa.int32())),
                plc.binaryop.BinaryOperator.MUL,
                plc.types.DataType(plc.types.TypeId.INT32),
            )
            total_nanos = plc.binaryop.binary_operation(
                nanos,
                millis_as_nanos,
                plc.binaryop.BinaryOperator.ADD,
                plc.types.DataType(plc.types.TypeId.INT32),
            )
            total_nanos = plc.binaryop.binary_operation(
                total_nanos,
                micros_as_nanos,
                plc.binaryop.BinaryOperator.ADD,
                plc.types.DataType(plc.types.TypeId.INT32),
            )
            return Column(total_nanos)

        return Column(
            plc.datetime.extract_datetime_component(
                column.obj,
                self._COMPONENT_MAP[self.name],
            )
        )
