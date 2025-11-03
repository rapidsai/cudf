# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document TemporalFunction to remove noqa
# ruff: noqa: D101
"""DSL nodes for datetime operations."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from typing_extensions import Self

    from polars.polars import _expr_nodes as pl_expr

    from cudf_polars.containers import DataFrame, DataType

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
        DaysInMonth = auto()
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
        Replace = auto()
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

    _valid_ops: ClassVar[set[Name]] = {
        *_COMPONENT_MAP.keys(),
        Name.IsLeapYear,
        Name.OrdinalDay,
        Name.ToString,
        Name.Week,
        Name.IsoYear,
        Name.MonthStart,
        Name.MonthEnd,
        Name.CastTimeUnit,
    }

    def __init__(
        self,
        dtype: DataType,
        name: TemporalFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = True
        if self.name not in self._valid_ops:
            raise NotImplementedError(f"Temporal function {self.name}")

        if self.name is TemporalFunction.Name.ToString and plc.traits.is_duration(
            self.children[0].dtype.plc_type
        ):
            raise NotImplementedError("ToString is not supported on duration types")

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = [child.evaluate(df, context=context) for child in self.children]
        (column,) = columns
        if self.name is TemporalFunction.Name.CastTimeUnit:
            return Column(
                plc.unary.cast(column.obj, self.dtype.plc_type, stream=df.stream),
                dtype=self.dtype,
            )
        if self.name == TemporalFunction.Name.ToString:
            return Column(
                plc.strings.convert.convert_datetime.from_timestamps(
                    column.obj,
                    self.options[0],
                    plc.Column.from_iterable_of_py(
                        [], dtype=self.dtype.plc_type, stream=df.stream
                    ),
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        if self.name is TemporalFunction.Name.Week:
            result = plc.strings.convert.convert_integers.to_integers(
                plc.strings.convert.convert_datetime.from_timestamps(
                    column.obj,
                    format="%V",
                    input_strings_names=plc.Column.from_iterable_of_py(
                        [], dtype=plc.DataType(plc.TypeId.STRING), stream=df.stream
                    ),
                    stream=df.stream,
                ),
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(result, dtype=self.dtype)
        if self.name is TemporalFunction.Name.IsoYear:
            result = plc.strings.convert.convert_integers.to_integers(
                plc.strings.convert.convert_datetime.from_timestamps(
                    column.obj,
                    format="%G",
                    input_strings_names=plc.Column.from_iterable_of_py(
                        [], dtype=plc.DataType(plc.TypeId.STRING), stream=df.stream
                    ),
                    stream=df.stream,
                ),
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(result, dtype=self.dtype)
        if self.name is TemporalFunction.Name.MonthStart:
            ends = plc.datetime.last_day_of_month(column.obj, stream=df.stream)
            days_to_subtract = plc.datetime.days_in_month(column.obj, stream=df.stream)
            # must subtract 1 to avoid rolling over to the previous month
            days_to_subtract = plc.binaryop.binary_operation(
                days_to_subtract,
                plc.Scalar.from_py(1, plc.DataType(plc.TypeId.INT32), stream=df.stream),
                plc.binaryop.BinaryOperator.SUB,
                plc.DataType(plc.TypeId.DURATION_DAYS),
                stream=df.stream,
            )
            result = plc.binaryop.binary_operation(
                ends,
                days_to_subtract,
                plc.binaryop.BinaryOperator.SUB,
                self.dtype.plc_type,
                stream=df.stream,
            )

            return Column(result, dtype=self.dtype)
        if self.name is TemporalFunction.Name.MonthEnd:
            return Column(
                plc.unary.cast(
                    plc.datetime.last_day_of_month(column.obj, stream=df.stream),
                    self.dtype.plc_type,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        if self.name is TemporalFunction.Name.IsLeapYear:
            return Column(
                plc.datetime.is_leap_year(column.obj, stream=df.stream),
                dtype=self.dtype,
            )
        if self.name is TemporalFunction.Name.OrdinalDay:
            return Column(
                plc.datetime.day_of_year(column.obj, stream=df.stream), dtype=self.dtype
            )
        if self.name is TemporalFunction.Name.Microsecond:
            millis = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MILLISECOND, stream=df.stream
            )
            micros = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MICROSECOND, stream=df.stream
            )
            millis_as_micros = plc.binaryop.binary_operation(
                millis,
                plc.Scalar.from_py(
                    1_000, plc.DataType(plc.TypeId.INT32), stream=df.stream
                ),
                plc.binaryop.BinaryOperator.MUL,
                self.dtype.plc_type,
                stream=df.stream,
            )
            total_micros = plc.binaryop.binary_operation(
                micros,
                millis_as_micros,
                plc.binaryop.BinaryOperator.ADD,
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(total_micros, dtype=self.dtype)
        elif self.name is TemporalFunction.Name.Nanosecond:
            millis = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MILLISECOND, stream=df.stream
            )
            micros = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MICROSECOND, stream=df.stream
            )
            nanos = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.NANOSECOND, stream=df.stream
            )
            millis_as_nanos = plc.binaryop.binary_operation(
                millis,
                plc.Scalar.from_py(
                    1_000_000, plc.DataType(plc.TypeId.INT32), stream=df.stream
                ),
                plc.binaryop.BinaryOperator.MUL,
                self.dtype.plc_type,
                stream=df.stream,
            )
            micros_as_nanos = plc.binaryop.binary_operation(
                micros,
                plc.Scalar.from_py(
                    1_000, plc.DataType(plc.TypeId.INT32), stream=df.stream
                ),
                plc.binaryop.BinaryOperator.MUL,
                self.dtype.plc_type,
                stream=df.stream,
            )
            total_nanos = plc.binaryop.binary_operation(
                nanos,
                millis_as_nanos,
                plc.binaryop.BinaryOperator.ADD,
                self.dtype.plc_type,
                stream=df.stream,
            )
            total_nanos = plc.binaryop.binary_operation(
                total_nanos,
                micros_as_nanos,
                plc.binaryop.BinaryOperator.ADD,
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(total_nanos, dtype=self.dtype)

        return Column(
            plc.datetime.extract_datetime_component(
                column.obj,
                self._COMPONENT_MAP[self.name],
                stream=df.stream,
            ),
            dtype=self.dtype,
        )
