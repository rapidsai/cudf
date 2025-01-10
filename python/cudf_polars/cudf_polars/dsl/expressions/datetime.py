# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for datetime operations."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    from polars.polars import _expr_nodes as pl_expr

    from cudf_polars.containers import DataFrame

__all__ = ["TemporalFunction"]


class TemporalFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `TemporalFunction`."""

        BaseUtcOffset = auto()
        CastTimeUnit = auto()
        Century = auto()
        Combine = auto()
        ConvertTimeZone = auto()
        DSTOffset = auto()
        Date = auto()
        Datetime = auto()
        DatetimeFunction = auto()
        Day = auto()
        Duration = auto()
        Hour = auto()
        IsLeapYear = auto()
        IsoYear = auto()
        Microsecond = auto()
        Millennium = auto()
        Millisecond = auto()
        Minute = auto()
        Month = auto()
        MonthEnd = auto()
        MonthStart = auto()
        Nanosecond = auto()
        OffsetBy = auto()
        OrdinalDay = auto()
        Quarter = auto()
        ReplaceTimeZone = auto()
        Round = auto()
        Second = auto()
        Time = auto()
        TimeStamp = auto()
        ToString = auto()
        TotalDays = auto()
        TotalHours = auto()
        TotalMicroseconds = auto()
        TotalMilliseconds = auto()
        TotalMinutes = auto()
        TotalNanoseconds = auto()
        TotalSeconds = auto()
        Truncate = auto()
        Week = auto()
        WeekDay = auto()
        WithTimeUnit = auto()
        Year = auto()

        @classmethod
        def from_polars(cls, obj: pl_expr.TemporalFunction) -> Self:
            """Convert from polars' `TemporalFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "TemporalFunction":
                raise ValueError("TemporalFunction required")
            return getattr(cls, name)

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")
    _COMPONENT_MAP: ClassVar[dict[Name, plc.datetime.DatetimeComponent]] = {
        Name.Year: plc.datetime.DatetimeComponent.YEAR,
        Name.Month: plc.datetime.DatetimeComponent.MONTH,
        Name.Day: plc.datetime.DatetimeComponent.DAY,
        Name.WeekDay: plc.datetime.DatetimeComponent.WEEKDAY,
        Name.Hour: plc.datetime.DatetimeComponent.HOUR,
        Name.Minute: plc.datetime.DatetimeComponent.MINUTE,
        Name.Second: plc.datetime.DatetimeComponent.SECOND,
        Name.Millisecond: plc.datetime.DatetimeComponent.MILLISECOND,
        Name.Microsecond: plc.datetime.DatetimeComponent.MICROSECOND,
        Name.Nanosecond: plc.datetime.DatetimeComponent.NANOSECOND,
    }

    def __init__(
        self,
        dtype: plc.DataType,
        name: TemporalFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = True
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
        if self.name is TemporalFunction.Name.Microsecond:
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
        elif self.name is TemporalFunction.Name.Nanosecond:
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
