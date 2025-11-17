# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import calendar
import functools
import locale
import re
import warnings
from locale import nl_langinfo
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.core._internals import binaryop
from cudf.core._internals.timezones import (
    check_ambiguous_and_nonexistent,
    get_compatible_timezone,
    get_tz_data,
)
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column.column import ColumnBase, as_column
from cudf.core.column.temporal_base import TemporalBaseColumn
from cudf.utils.dtypes import (
    _get_base_dtype,
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    get_dtype_of_same_kind,
    get_dtype_of_same_type,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf._typing import (
        ColumnBinaryOperand,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.string import StringColumn

# nanoseconds per time_unit
_dtype_to_format_conversion = {
    "datetime64[ns]": "%Y-%m-%d %H:%M:%S.%9f",
    "datetime64[us]": "%Y-%m-%d %H:%M:%S.%6f",
    "datetime64[ms]": "%Y-%m-%d %H:%M:%S.%3f",
    "datetime64[s]": "%Y-%m-%d %H:%M:%S",
}

_DATETIME_SPECIAL_FORMATS = {
    "%b",
    "%B",
    "%A",
    "%a",
}


def _resolve_binop_resolution(
    left_unit: Literal["s", "ms", "us", "ns"],
    right_unit: Literal["s", "ms", "us", "ns"],
) -> Literal["s", "ms", "us", "ns"]:
    units: list[Literal["s", "ms", "us", "ns"]] = ["s", "ms", "us", "ns"]
    left_idx = units.index(left_unit)
    right_idx = units.index(right_unit)
    return units[max(left_idx, right_idx)]


class DatetimeColumn(TemporalBaseColumn):
    """
    A Column implementation for Date-time types.

    Parameters
    ----------
    data : Buffer
        The datetime values
    dtype : np.dtype
        The data type
    mask : Buffer; optional
        The validity mask
    """

    _NP_SCALAR = np.datetime64
    _PD_SCALAR = pd.Timestamp
    _VALID_BINARY_OPERATIONS = {
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__add__",
        "__sub__",
        "__radd__",
        "__rsub__",
    }
    _VALID_PLC_TYPES = {
        plc.TypeId.TIMESTAMP_SECONDS,
        plc.TypeId.TIMESTAMP_MILLISECONDS,
        plc.TypeId.TIMESTAMP_MICROSECONDS,
        plc.TypeId.TIMESTAMP_NANOSECONDS,
    }

    def __init__(
        self,
        plc_column: plc.Column,
        size: int,
        dtype: np.dtype | pd.DatetimeTZDtype,
        offset: int,
        null_count: int,
        exposed: bool,
    ) -> None:
        dtype = self._validate_dtype_instance(dtype)
        super().__init__(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )

    def _clear_cache(self) -> None:
        super()._clear_cache()
        attrs = (
            "days_in_month",
            "is_year_start",
            "is_leap_year",
            "is_year_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_month_start",
            "is_month_end",
            "day_of_year",
            "weekday",
            "nanosecond",
            "microsecond",
            "millisecond",
            "second",
            "minute",
            "hour",
            "day",
            "month",
            "year",
            "quarter",
            "time_unit",
        )
        for attr in attrs:
            try:
                delattr(self, attr)
            except AttributeError:
                # attr was not called yet, so ignore.
                pass

    def _scan(self, op: str) -> ColumnBase:
        if op not in {"cummin", "cummax"}:
            raise TypeError(
                f"Accumulation {op} not supported for {self.dtype}"
            )
        return self.scan(op.replace("cum", ""), True)._with_type_metadata(
            self.dtype
        )

    @staticmethod
    def _validate_dtype_instance(dtype: np.dtype) -> np.dtype:
        if (
            cudf.get_option("mode.pandas_compatible") and not dtype.kind == "M"
        ) or (
            not cudf.get_option("mode.pandas_compatible")
            and not (isinstance(dtype, np.dtype) and dtype.kind == "M")
        ):
            raise ValueError(f"dtype must be a datetime, got {dtype}")
        return dtype

    def __contains__(self, item: ScalarLike) -> bool:
        try:
            ts = self._PD_SCALAR(item).as_unit(self.time_unit)
        except Exception:
            # pandas can raise a variety of errors
            # item cannot exist in self.
            return False
        if ts.tzinfo is None and isinstance(self.dtype, pd.DatetimeTZDtype):
            return False
        elif ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return super().__contains__(ts.to_numpy())

    @functools.cached_property
    @acquire_spill_lock()
    def quarter(self) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.datetime.extract_quarter(self.to_pylibcudf(mode="read"))
        )

    @functools.cached_property
    def year(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.YEAR)

    @functools.cached_property
    def month(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MONTH)

    @functools.cached_property
    def day(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.DAY)

    @functools.cached_property
    def hour(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.HOUR)

    @functools.cached_property
    def minute(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MINUTE)

    @functools.cached_property
    def second(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.SECOND)

    @functools.cached_property
    def millisecond(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MILLISECOND)

    @functools.cached_property
    def microsecond(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MICROSECOND)

    @functools.cached_property
    def nanosecond(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.NANOSECOND)

    @functools.cached_property
    def weekday(self) -> ColumnBase:
        # pandas counts Monday-Sunday as 0-6
        # while libcudf counts Monday-Sunday as 1-7
        result = self._get_dt_field(plc.datetime.DatetimeComponent.WEEKDAY)
        return result - result.dtype.type(1)

    @functools.cached_property
    @acquire_spill_lock()
    def day_of_year(self) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.datetime.day_of_year(self.to_pylibcudf(mode="read"))
        )

    @functools.cached_property
    def is_month_start(self) -> ColumnBase:
        return (self.day == 1).fillna(False)

    @functools.cached_property
    def is_month_end(self) -> ColumnBase:
        with acquire_spill_lock():
            last_day_col = type(self).from_pylibcudf(
                plc.datetime.last_day_of_month(self.to_pylibcudf(mode="read"))
            )
        return (self.day == last_day_col.day).fillna(False)

    @functools.cached_property
    def is_quarter_end(self) -> ColumnBase:
        last_month = self.month.isin([3, 6, 9, 12])
        return (self.is_month_end & last_month).fillna(False)

    @functools.cached_property
    def is_quarter_start(self) -> ColumnBase:
        first_month = self.month.isin([1, 4, 7, 10])
        return (self.is_month_start & first_month).fillna(False)

    @functools.cached_property
    def is_year_end(self) -> ColumnBase:
        day_of_year = self.day_of_year
        leap_dates = self.is_leap_year

        leap = day_of_year == 366
        non_leap = day_of_year == 365
        return leap.copy_if_else(non_leap, leap_dates).fillna(False)

    @functools.cached_property
    @acquire_spill_lock()
    def is_leap_year(self) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.datetime.is_leap_year(self.to_pylibcudf(mode="read"))
        )

    @functools.cached_property
    def is_year_start(self) -> ColumnBase:
        return (self.day_of_year == 1).fillna(False)

    @functools.cached_property
    @acquire_spill_lock()
    def days_in_month(self) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.datetime.days_in_month(self.to_pylibcudf(mode="read"))
        )

    @functools.cached_property
    def day_of_week(self) -> ColumnBase:
        raise NotImplementedError("day_of_week is currently not implemented.")

    @functools.cached_property
    def tz(self):
        """
        Return the timezone.

        Returns
        -------
        datetime.tzinfo or None
            Returns None when the array is tz-naive.
        """
        if isinstance(self.dtype, pd.DatetimeTZDtype):
            return self.dtype.tz
        return None

    @functools.cached_property
    def time_unit(self) -> str:
        return np.datetime_data(self.dtype)[0]

    @functools.cached_property
    def freq(self) -> str | None:
        raise NotImplementedError("freq is not yet implemented.")

    @functools.cached_property
    def date(self):
        raise NotImplementedError("date is not yet implemented.")

    @functools.cached_property
    def time(self):
        raise NotImplementedError("time is not yet implemented.")

    @functools.cached_property
    def timetz(self):
        raise NotImplementedError("timetz is not yet implemented.")

    @functools.cached_property
    def is_normalized(self) -> bool:
        raise NotImplementedError(
            "is_normalized is currently not implemented."
        )

    def to_julian_date(self) -> ColumnBase:
        raise NotImplementedError(
            "to_julian_date is currently not implemented."
        )

    def normalize(self) -> ColumnBase:
        raise NotImplementedError("normalize is currently not implemented.")

    @acquire_spill_lock()
    def _get_dt_field(
        self, field: plc.datetime.DatetimeComponent
    ) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.datetime.extract_datetime_component(
                self.to_pylibcudf(mode="read"),
                field,
            )
        )

    def _get_field_names(
        self,
        field: Literal["month", "weekday"],
        labels: list[str],
        locale: str | None = None,
    ) -> ColumnBase:
        if locale is not None:
            raise NotImplementedError(
                "Setting a locale is currently not supported. "
                "Results will be returned in your current locale."
            )
        col_labels = as_column(labels)
        indices = getattr(self, field)
        has_nulls = indices.has_nulls()
        if has_nulls:
            indices = indices.fillna(len(col_labels))
        return col_labels.take(indices, nullify=True, check_bounds=has_nulls)

    def get_day_names(self, locale: str | None = None) -> ColumnBase:
        return self._get_field_names(
            "weekday", list(calendar.day_name), locale=locale
        )

    def get_month_names(self, locale: str | None = None) -> ColumnBase:
        return self._get_field_names(
            "month", list(calendar.month_name), locale=locale
        )

    def _round_dt(
        self,
        round_func: Callable[
            [plc.Column, plc.datetime.RoundingFrequency], plc.Column
        ],
        freq: str,
    ) -> ColumnBase:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.resolution_string.html
        old_to_new_freq_map = {
            "H": "h",
            "N": "ns",
            "T": "min",
            "L": "ms",
            "U": "us",
            "S": "s",
        }
        if freq in old_to_new_freq_map:
            warnings.warn(
                f"{freq} is deprecated and will be "
                "removed in a future version, please use "
                f"{old_to_new_freq_map[freq]} instead.",
                FutureWarning,
            )
            freq = old_to_new_freq_map[freq]
        rounding_fequency_map = {
            "D": plc.datetime.RoundingFrequency.DAY,
            "h": plc.datetime.RoundingFrequency.HOUR,
            "min": plc.datetime.RoundingFrequency.MINUTE,
            "s": plc.datetime.RoundingFrequency.SECOND,
            "ms": plc.datetime.RoundingFrequency.MILLISECOND,
            "us": plc.datetime.RoundingFrequency.MICROSECOND,
            "ns": plc.datetime.RoundingFrequency.NANOSECOND,
        }
        if (plc_freq := rounding_fequency_map.get(freq)) is None:
            raise ValueError(f"Invalid resolution: '{freq}'")

        with acquire_spill_lock():
            return type(self).from_pylibcudf(
                round_func(
                    self.to_pylibcudf(mode="read"),
                    plc_freq,
                )
            )

    def ceil(self, freq: str) -> ColumnBase:
        return self._round_dt(plc.datetime.ceil_datetimes, freq)

    def floor(self, freq: str) -> ColumnBase:
        return self._round_dt(plc.datetime.floor_datetimes, freq)

    def round(self, freq: str) -> ColumnBase:
        return self._round_dt(plc.datetime.round_datetimes, freq)

    def isocalendar(self) -> dict[str, ColumnBase]:
        return {
            field: self.strftime(format=directive).astype(np.dtype(np.uint32))
            for field, directive in zip(
                ["year", "week", "day"], ["%G", "%V", "%u"], strict=True
            )
        }

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        if dtype == self.dtype:
            return self
        elif isinstance(dtype, pd.DatetimeTZDtype):
            raise TypeError(
                "Cannot use .astype to convert from timezone-naive dtype to timezone-aware dtype. "
                "Use tz_localize instead."
            )
        return self.cast(dtype=dtype)  # type: ignore[return-value]

    def as_timedelta_column(self, dtype: np.dtype) -> None:  # type: ignore[override]
        raise TypeError(
            f"cannot astype a datetimelike from {self.dtype} to {dtype}"
        )

    @functools.cached_property
    def _strftime_names(self) -> plc.Column:
        """Strftime names for %A, %a, %B, %b"""
        return plc.Column.from_iterable_of_py(
            [
                nl_langinfo(loc)
                for loc in (
                    locale.AM_STR,
                    locale.PM_STR,
                    locale.DAY_1,
                    locale.DAY_2,
                    locale.DAY_3,
                    locale.DAY_4,
                    locale.DAY_5,
                    locale.DAY_6,
                    locale.DAY_7,
                    locale.ABDAY_1,
                    locale.ABDAY_2,
                    locale.ABDAY_3,
                    locale.ABDAY_4,
                    locale.ABDAY_5,
                    locale.ABDAY_6,
                    locale.ABDAY_7,
                    locale.MON_1,
                    locale.MON_2,
                    locale.MON_3,
                    locale.MON_4,
                    locale.MON_5,
                    locale.MON_6,
                    locale.MON_7,
                    locale.MON_8,
                    locale.MON_9,
                    locale.MON_10,
                    locale.MON_11,
                    locale.MON_12,
                    locale.ABMON_1,
                    locale.ABMON_2,
                    locale.ABMON_3,
                    locale.ABMON_4,
                    locale.ABMON_5,
                    locale.ABMON_6,
                    locale.ABMON_7,
                    locale.ABMON_8,
                    locale.ABMON_9,
                    locale.ABMON_10,
                    locale.ABMON_11,
                    locale.ABMON_12,
                )
            ]
        )

    def strftime(self, format: str) -> StringColumn:
        if len(self) == 0:
            return super().strftime(format)
        if re.search("%[aAbB]", format):
            names = self._strftime_names
        else:
            names = plc.Column.from_scalar(
                plc.Scalar.from_py(None, plc.DataType(plc.TypeId.STRING)), 0
            )
        with acquire_spill_lock():
            return type(self).from_pylibcudf(  # type: ignore[return-value]
                plc.strings.convert.convert_datetime.from_timestamps(
                    self.to_pylibcudf(mode="read"),
                    format,
                    names,
                )
            )

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        format = _dtype_to_format_conversion.get(
            self.dtype.name, "%Y-%m-%d %H:%M:%S"
        )
        if cudf.get_option("mode.pandas_compatible"):
            if isinstance(dtype, np.dtype) and dtype.kind == "O":
                raise TypeError(
                    f"Cannot astype a datetimelike from {self.dtype} to {dtype}"
                )
            if format.endswith("f"):
                sub_second_res_len = 3
            else:
                sub_second_res_len = 0

            has_nanos = self.time_unit == "ns" and self.nanosecond.any()
            has_micros = (
                self.time_unit in {"ns", "us"} and self.microsecond.any()
            )
            has_millis = (
                self.time_unit in {"ns", "us", "ms"} and self.millisecond.any()
            )
            has_seconds = self.second.any()
            has_minutes = self.minute.any()
            has_hours = self.hour.any()
            if sub_second_res_len:
                if has_nanos:
                    # format should be intact and rest of the
                    # following conditions shouldn't execute.
                    pass
                elif has_micros:
                    format = format[:-sub_second_res_len] + "%6f"
                elif has_millis:
                    format = format[:-sub_second_res_len] + "%3f"
                elif has_seconds or has_minutes or has_hours:
                    format = format[:-4]
                else:
                    format = format.split(" ")[0]
            elif not (has_seconds or has_minutes or has_hours):
                format = format.split(" ")[0]
        return self.strftime(format)

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)
        if isinstance(other, cudf.DateOffset):
            return other._datetime_binop(self, op, reflect=reflect)
        other = self._normalize_binop_operand(other)
        if other is NotImplemented:
            return NotImplemented

        if reflect:
            lhs = other
            rhs = self
            if isinstance(lhs, pa.Scalar):
                lhs_unit = lhs.type.unit
                other_dtype = cudf_dtype_from_pa_type(lhs.type)
            else:
                lhs_unit = lhs.time_unit  # type: ignore[attr-defined]
                other_dtype = lhs.dtype
            rhs_unit = rhs.time_unit
        else:
            lhs = self
            rhs = other  # type: ignore[assignment]
            if isinstance(rhs, pa.Scalar):
                rhs_unit = rhs.type.unit
                other_dtype = cudf_dtype_from_pa_type(rhs.type)
            else:
                rhs_unit = rhs.time_unit
                other_dtype = rhs.dtype
            lhs_unit = lhs.time_unit

        other_is_timedelta = other_dtype.kind == "m"
        other_is_datetime64 = other_dtype.kind == "M"

        out_dtype = None

        if (
            op
            in {
                "__ne__",
                "__lt__",
                "__gt__",
                "__le__",
                "__ge__",
            }
            and other_is_datetime64
        ):
            out_dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))
        elif op == "__add__" and other_is_timedelta:
            # The only thing we can add to a datetime is a timedelta. This
            # operation is symmetric, i.e. we allow `datetime + timedelta` or
            # `timedelta + datetime`. Both result in DatetimeColumns.
            out_dtype = get_dtype_of_same_kind(
                self.dtype,
                np.dtype(
                    f"datetime64[{_resolve_binop_resolution(lhs_unit, rhs_unit)}]"  # type: ignore[arg-type]
                ),
            )
        elif op == "__sub__":
            # Subtracting a datetime from a datetime results in a timedelta.
            if other_is_datetime64:
                out_dtype = get_dtype_of_same_kind(
                    self.dtype,
                    np.dtype(
                        f"timedelta64[{_resolve_binop_resolution(lhs_unit, rhs_unit)}]"  # type: ignore[arg-type]
                    ),
                )
            # We can subtract a timedelta from a datetime, but not vice versa.
            # Not only is subtraction antisymmetric (as is normal), it is only
            # well-defined if this operation was not invoked via reflection.
            elif other_is_timedelta and not reflect:
                out_dtype = get_dtype_of_same_kind(
                    self.dtype,
                    np.dtype(
                        f"datetime64[{_resolve_binop_resolution(lhs_unit, rhs_unit)}]"  # type: ignore[arg-type]
                    ),
                )
        elif op in {
            "__eq__",
            "__ne__",
            "NULL_EQUALS",
            "NULL_NOT_EQUALS",
        }:
            out_dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))
            if isinstance(other, ColumnBase) and not isinstance(
                other, DatetimeColumn
            ):
                fill_value = op in ("__ne__", "NULL_NOT_EQUALS")
                result = self._all_bools_with_nulls(
                    other, bool_fill_value=fill_value
                )
                if cudf.get_option("mode.pandas_compatible"):
                    result = result.fillna(fill_value)
                return result

        if out_dtype is None:
            return NotImplemented

        lhs_binop: plc.Scalar | ColumnBase = (
            pa_scalar_to_plc_scalar(lhs) if isinstance(lhs, pa.Scalar) else lhs
        )
        rhs_binop: plc.Scalar | ColumnBase = (
            pa_scalar_to_plc_scalar(rhs) if isinstance(rhs, pa.Scalar) else rhs
        )

        result_col = binaryop.binaryop(lhs_binop, rhs_binop, op, out_dtype)
        if out_dtype.kind != "b" and op == "__add__":
            return result_col
        elif (
            cudf.get_option("mode.pandas_compatible") and out_dtype.kind == "b"
        ):
            return result_col.fillna(op == "__ne__")
        else:
            return result_col

    def _with_type_metadata(self, dtype: DtypeObj) -> DatetimeColumn:
        if isinstance(dtype, pd.DatetimeTZDtype):
            return DatetimeTZColumn(
                plc_column=self.plc_column,
                size=self.size,
                dtype=dtype,
                offset=self.offset,
                null_count=self.null_count,
                exposed=False,
            )
        if cudf.get_option("mode.pandas_compatible"):
            self._dtype = get_dtype_of_same_type(dtype, self.dtype)

        return self

    def _find_ambiguous_and_nonexistent(
        self, zone_name: str
    ) -> tuple[NumericalColumn, NumericalColumn] | tuple[bool, bool]:
        """
        Recognize ambiguous and nonexistent timestamps for the given timezone.

        Returns a tuple of columns, both of "bool" dtype and of the same
        size as `self`, that respectively indicate ambiguous and
        nonexistent timestamps in `self` with the value `True`.

        Ambiguous and/or nonexistent timestamps are only possible if any
        transitions occur in the time zone database for the given timezone.
        If no transitions occur, the tuple `(False, False)` is returned.
        """
        transition_times, offsets = get_tz_data(zone_name)
        offsets = offsets.astype(np.dtype(f"timedelta64[{self.time_unit}]"))  # type: ignore[assignment]

        if len(offsets) == 1:  # no transitions
            return False, False

        transition_times, offsets, old_offsets = (
            transition_times.slice(1, len(transition_times)),
            offsets.slice(1, len(offsets)),
            offsets.slice(0, len(offsets) - 1),
        )

        # Assume we have two clocks at the moment of transition:
        # - Clock 1 is turned forward or backwards correctly
        # - Clock 2 makes no changes
        clock_1 = transition_times + offsets
        clock_2 = transition_times + old_offsets

        # At the start of an ambiguous time period, Clock 1 (which has
        # been turned back) reads less than Clock 2:
        cond = clock_1 < clock_2
        ambiguous_begin = clock_1.apply_boolean_mask(cond)

        # The end of an ambiguous time period is what Clock 2 reads at
        # the moment of transition:
        ambiguous_end = clock_2.apply_boolean_mask(cond)
        ambiguous = self.label_bins(
            left_edge=ambiguous_begin,
            left_inclusive=True,
            right_edge=ambiguous_end,
            right_inclusive=False,
        ).notnull()

        # At the start of a non-existent time period, Clock 2 reads less
        # than Clock 1 (which has been turned forward):
        cond = clock_1 > clock_2
        nonexistent_begin = clock_2.apply_boolean_mask(cond)

        # The end of the non-existent time period is what Clock 1 reads
        # at the moment of transition:
        nonexistent_end = clock_1.apply_boolean_mask(cond)
        nonexistent = self.label_bins(
            left_edge=nonexistent_begin,
            left_inclusive=True,
            right_edge=nonexistent_end,
            right_inclusive=False,
        ).notnull()

        return ambiguous, nonexistent  # type: ignore[return-value]

    def tz_localize(
        self,
        tz: str | None,
        ambiguous: Literal["NaT"] = "NaT",
        nonexistent: Literal["NaT"] = "NaT",
    ) -> DatetimeColumn:
        if tz is None:
            return self.copy()
        ambiguous, nonexistent = check_ambiguous_and_nonexistent(
            ambiguous, nonexistent
        )
        dtype = get_compatible_timezone(pd.DatetimeTZDtype(self.time_unit, tz))
        tzname = dtype.tz.key
        ambiguous_col, nonexistent_col = self._find_ambiguous_and_nonexistent(
            tzname
        )
        localized = self._scatter_by_column(
            self.isnull() | (ambiguous_col | nonexistent_col),
            pa_scalar_to_plc_scalar(
                pa.scalar(None, type=cudf_dtype_to_pa_type(self.dtype))
            ),
        )

        transition_times, offsets = get_tz_data(tzname)
        transition_times_local = (transition_times + offsets).astype(
            localized.dtype
        )
        indices = (
            transition_times_local.searchsorted(localized, side="right") - 1
        )
        offsets_to_utc = offsets.take(indices, nullify=True)
        gmt_data = localized - offsets_to_utc
        return gmt_data._with_type_metadata(dtype)

    def tz_convert(self, tz: str | None) -> DatetimeColumn:
        raise TypeError(
            "Cannot convert tz-naive timestamps, use tz_localize to localize"
        )


class DatetimeTZColumn(DatetimeColumn):
    def _clear_cache(self) -> None:
        super()._clear_cache()
        try:
            del self._local_time
        except AttributeError:
            pass

    @staticmethod
    def _validate_dtype_instance(
        dtype: pd.DatetimeTZDtype,
    ) -> pd.DatetimeTZDtype:
        if not isinstance(dtype, pd.DatetimeTZDtype):
            raise ValueError("dtype must be a pandas.DatetimeTZDtype")
        return get_compatible_timezone(dtype)

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if (
            arrow_type
            or nullable
            or (
                cudf.get_option("mode.pandas_compatible")
                and isinstance(self.dtype, pd.ArrowDtype)
            )
        ):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        else:
            return self._local_time.to_pandas().tz_localize(
                self.dtype.tz,  # type: ignore[union-attr]
                ambiguous="NaT",
                nonexistent="NaT",
            )

    def to_arrow(self) -> pa.Array:
        # Cast to expected timestamp array type for assume_timezone
        local_array = cast(pa.TimestampArray, self._local_time.to_arrow())
        return pa.compute.assume_timezone(local_array, str(self.dtype.tz))  # type: ignore[union-attr]

    @functools.cached_property
    def time_unit(self) -> str:
        return self.dtype.unit  # type: ignore[union-attr]

    @property
    def _utc_time(self) -> DatetimeColumn:
        """Return UTC time as naive timestamps."""
        return DatetimeColumn(
            plc_column=self.plc_column,
            size=self.size,
            dtype=_get_base_dtype(self.dtype),
            offset=self.offset,
            null_count=self.null_count,
            exposed=False,
        )

    @functools.cached_property
    def _local_time(self) -> DatetimeColumn:
        """Return the local time as naive timestamps."""
        transition_times, offsets = get_tz_data(str(self.dtype.tz))  # type: ignore[union-attr]
        base_dtype = _get_base_dtype(self.dtype)
        indices = (
            transition_times.astype(base_dtype).searchsorted(
                self.astype(base_dtype), side="right"
            )
            - 1
        )
        offsets_from_utc = offsets.take(indices, nullify=True)
        return self + offsets_from_utc

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        return self._local_time.as_string_column(dtype)

    def as_datetime_column(
        self, dtype: np.dtype | pd.DatetimeTZDtype
    ) -> DatetimeColumn:
        if isinstance(dtype, pd.DatetimeTZDtype) and dtype != self.dtype:
            if dtype.unit != self.time_unit:
                # TODO: Doesn't check that new unit is valid.
                casted = self._with_type_metadata(dtype)
            else:
                casted = self
            return casted.tz_convert(str(dtype.tz))
        return super().as_datetime_column(dtype)

    @acquire_spill_lock()
    def _get_dt_field(
        self, field: plc.datetime.DatetimeComponent
    ) -> ColumnBase:
        return type(self).from_pylibcudf(
            plc.datetime.extract_datetime_component(
                self._local_time.to_pylibcudf(mode="read"),
                field,
            )
        )

    def __repr__(self) -> str:
        # Arrow prints the UTC timestamps, but we want to print the
        # local timestamps:
        arr = self._local_time.to_arrow().cast(
            pa.timestamp(self.dtype.unit, str(self.dtype.tz))  # type: ignore[union-attr]
        )
        return (
            f"{object.__repr__(self)}\n{arr.to_string()}\ndtype: {self.dtype}"
        )

    def tz_localize(
        self,
        tz: str | None,
        ambiguous: Literal["NaT"] = "NaT",
        nonexistent: Literal["NaT"] = "NaT",
    ) -> DatetimeColumn:
        if tz is None:
            return self._local_time
        ambiguous, nonexistent = check_ambiguous_and_nonexistent(
            ambiguous, nonexistent
        )
        raise ValueError(
            "Already localized. "
            "Use `tz_convert` to convert between time zones."
        )

    def tz_convert(self, tz: str | None) -> DatetimeColumn:
        if tz is None:
            return self._utc_time
        elif tz == str(self.dtype.tz):  # type: ignore[union-attr]
            return self.copy()
        utc_time = self._utc_time
        return utc_time._with_type_metadata(
            pd.DatetimeTZDtype(self.time_unit, tz)
        )
