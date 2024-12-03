# Copyright (c) 2019-2024, NVIDIA CORPORATION.

from __future__ import annotations

import calendar
import datetime
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
import cudf.core.column.column as column
import cudf.core.column.string as string
from cudf import _lib as libcudf
from cudf.core._compat import PANDAS_GE_220
from cudf.core._internals import unary
from cudf.core._internals.search import search_sorted
from cudf.core._internals.timezones import (
    check_ambiguous_and_nonexistent,
    get_compatible_timezone,
    get_tz_data,
)
from cudf.core.buffer import Buffer, acquire_spill_lock
from cudf.core.column.column import ColumnBase, as_column
from cudf.core.column.timedelta import _unit_to_nanoseconds_conversion
from cudf.utils.dtypes import _get_base_dtype
from cudf.utils.utils import (
    _all_bools_with_nulls,
    _datetime_timedelta_find_and_replace,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cudf._typing import (
        ColumnBinaryOperand,
        DatetimeLikeScalar,
        Dtype,
        ScalarLike,
    )
    from cudf.core.column.numerical import NumericalColumn

if PANDAS_GE_220:
    _guess_datetime_format = pd.tseries.api.guess_datetime_format
else:
    _guess_datetime_format = pd.core.tools.datetimes.guess_datetime_format

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

_DATETIME_NAMES = [
    nl_langinfo(locale.AM_STR),  # type: ignore
    nl_langinfo(locale.PM_STR),  # type: ignore
    nl_langinfo(locale.DAY_1),
    nl_langinfo(locale.DAY_2),
    nl_langinfo(locale.DAY_3),
    nl_langinfo(locale.DAY_4),
    nl_langinfo(locale.DAY_5),
    nl_langinfo(locale.DAY_6),
    nl_langinfo(locale.DAY_7),
    nl_langinfo(locale.ABDAY_1),
    nl_langinfo(locale.ABDAY_2),
    nl_langinfo(locale.ABDAY_3),
    nl_langinfo(locale.ABDAY_4),
    nl_langinfo(locale.ABDAY_5),
    nl_langinfo(locale.ABDAY_6),
    nl_langinfo(locale.ABDAY_7),
    nl_langinfo(locale.MON_1),
    nl_langinfo(locale.MON_2),
    nl_langinfo(locale.MON_3),
    nl_langinfo(locale.MON_4),
    nl_langinfo(locale.MON_5),
    nl_langinfo(locale.MON_6),
    nl_langinfo(locale.MON_7),
    nl_langinfo(locale.MON_8),
    nl_langinfo(locale.MON_9),
    nl_langinfo(locale.MON_10),
    nl_langinfo(locale.MON_11),
    nl_langinfo(locale.MON_12),
    nl_langinfo(locale.ABMON_1),
    nl_langinfo(locale.ABMON_2),
    nl_langinfo(locale.ABMON_3),
    nl_langinfo(locale.ABMON_4),
    nl_langinfo(locale.ABMON_5),
    nl_langinfo(locale.ABMON_6),
    nl_langinfo(locale.ABMON_7),
    nl_langinfo(locale.ABMON_8),
    nl_langinfo(locale.ABMON_9),
    nl_langinfo(locale.ABMON_10),
    nl_langinfo(locale.ABMON_11),
    nl_langinfo(locale.ABMON_12),
]


def infer_format(element: str, **kwargs) -> str:
    """
    Infers datetime format from a string, also takes cares for `ms` and `ns`
    """
    if not cudf.get_option("mode.pandas_compatible"):
        # We allow "Z" but don't localize it to datetime64[ns, UTC] type (yet)
        element = element.replace("Z", "")
    fmt = _guess_datetime_format(element, **kwargs)

    if fmt is not None:
        if "%z" in fmt or "%Z" in fmt:
            raise NotImplementedError(
                "cuDF does not yet support timezone-aware datetimes"
            )
        if ".%f" not in fmt:
            # For context read:
            # https://github.com/pandas-dev/pandas/issues/52418
            # We cannot rely on format containing only %f
            # c++/libcudf expects .%3f, .%6f, .%9f
            # Logic below handles those cases well.
            return fmt

    element_parts = element.split(".")
    if len(element_parts) != 2:
        raise ValueError("Given date string not likely a datetime.")

    # There is possibility that the element is of following format
    # '00:00:03.333333 2016-01-01'
    second_parts = re.split(r"(\D+)", element_parts[1], maxsplit=1)
    subsecond_fmt = ".%" + str(len(second_parts[0])) + "f"

    first_part = _guess_datetime_format(element_parts[0], **kwargs)
    # For the case where first_part is '00:00:03'
    if first_part is None:
        tmp = "1970-01-01 " + element_parts[0]
        first_part = _guess_datetime_format(tmp, **kwargs).split(" ", 1)[1]
    if first_part is None:
        raise ValueError("Unable to infer the timestamp format from the data")

    if len(second_parts) > 1:
        # We may have a non-digit, timezone-like component
        # like Z, UTC-3, +01:00
        if any(re.search(r"\D", part) for part in second_parts):
            raise NotImplementedError(
                "cuDF does not yet support timezone-aware datetimes"
            )
        second_part = "".join(second_parts[1:])

        if len(second_part) > 1:
            # Only infer if second_parts is not an empty string.
            second_part = _guess_datetime_format(second_part, **kwargs)
    else:
        second_part = ""

    try:
        fmt = first_part + subsecond_fmt + second_part
    except Exception:
        raise ValueError("Unable to infer the timestamp format from the data")

    return fmt


def _resolve_mixed_dtypes(
    lhs: ColumnBinaryOperand, rhs: ColumnBinaryOperand, base_type: str
) -> Dtype:
    units = ["s", "ms", "us", "ns"]
    lhs_time_unit = cudf.utils.dtypes.get_time_unit(lhs)
    lhs_unit = units.index(lhs_time_unit)
    rhs_time_unit = cudf.utils.dtypes.get_time_unit(rhs)
    rhs_unit = units.index(rhs_time_unit)
    return cudf.dtype(f"{base_type}[{units[max(lhs_unit, rhs_unit)]}]")


class DatetimeColumn(column.ColumnBase):
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

    def __init__(
        self,
        data: Buffer,
        size: int | None,
        dtype: np.dtype | pd.DatetimeTZDtype,
        mask: Buffer | None = None,
        offset: int = 0,
        null_count: int | None = None,
        children: tuple = (),
    ):
        if not isinstance(data, Buffer):
            raise ValueError("data must be a Buffer.")
        dtype = self._validate_dtype_instance(dtype)
        if data.size % dtype.itemsize:
            raise ValueError("Buffer size must be divisible by element size")
        if size is None:
            size = data.size // dtype.itemsize
            size = size - offset
        if len(children) != 0:
            raise ValueError(f"{type(self).__name__} must have no children.")
        super().__init__(
            data=data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    @staticmethod
    def _validate_dtype_instance(dtype: np.dtype) -> np.dtype:
        if not (isinstance(dtype, np.dtype) and dtype.kind == "M"):
            raise ValueError("dtype must be a datetime, numpy dtype")
        return dtype

    def __contains__(self, item: ScalarLike) -> bool:
        try:
            ts = pd.Timestamp(item).as_unit(self.time_unit)
        except Exception:
            # pandas can raise a variety of errors
            # item cannot exist in self.
            return False
        if ts.tzinfo is None and isinstance(self.dtype, pd.DatetimeTZDtype):
            return False
        elif ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.to_numpy().astype("int64") in cast(
            "cudf.core.column.NumericalColumn", self.astype("int64")
        )

    @functools.cached_property
    def time_unit(self) -> str:
        return np.datetime_data(self.dtype)[0]

    @property
    def quarter(self) -> ColumnBase:
        with acquire_spill_lock():
            return type(self).from_pylibcudf(
                plc.datetime.extract_quarter(self.to_pylibcudf(mode="read"))
            )

    @property
    def year(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.YEAR)

    @property
    def month(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MONTH)

    @property
    def day(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.DAY)

    @property
    def hour(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.HOUR)

    @property
    def minute(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MINUTE)

    @property
    def second(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.SECOND)

    @property
    def millisecond(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MILLISECOND)

    @property
    def microsecond(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MICROSECOND)

    @property
    def nanosecond(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.NANOSECOND)

    @property
    def weekday(self) -> ColumnBase:
        # pandas counts Monday-Sunday as 0-6
        # while libcudf counts Monday-Sunday as 1-7
        result = self._get_dt_field(plc.datetime.DatetimeComponent.WEEKDAY)
        return result - result.dtype.type(1)

    @property
    def day_of_year(self) -> ColumnBase:
        with acquire_spill_lock():
            return type(self).from_pylibcudf(
                plc.datetime.day_of_year(self.to_pylibcudf(mode="read"))
            )

    @property
    def is_month_start(self) -> ColumnBase:
        return (self.day == 1).fillna(False)

    @property
    def is_month_end(self) -> ColumnBase:
        with acquire_spill_lock():
            last_day_col = type(self).from_pylibcudf(
                plc.datetime.last_day_of_month(self.to_pylibcudf(mode="read"))
            )
        return (self.day == last_day_col.day).fillna(False)  # type: ignore[attr-defined]

    @property
    def is_quarter_end(self) -> ColumnBase:
        last_month = self.month.isin([3, 6, 9, 12])
        return (self.is_month_end & last_month).fillna(False)

    @property
    def is_quarter_start(self) -> ColumnBase:
        first_month = self.month.isin([1, 4, 7, 10])
        return (self.is_month_start & first_month).fillna(False)

    @property
    def is_year_end(self) -> ColumnBase:
        day_of_year = self.day_of_year
        leap_dates = self.is_leap_year

        leap = day_of_year == cudf.Scalar(366)
        non_leap = day_of_year == cudf.Scalar(365)
        return libcudf.copying.copy_if_else(leap, non_leap, leap_dates).fillna(
            False
        )

    @property
    def is_leap_year(self) -> ColumnBase:
        with acquire_spill_lock():
            return type(self).from_pylibcudf(
                plc.datetime.is_leap_year(self.to_pylibcudf(mode="read"))
            )

    @property
    def is_year_start(self) -> ColumnBase:
        return (self.day_of_year == 1).fillna(False)

    @property
    def days_in_month(self) -> ColumnBase:
        with acquire_spill_lock():
            return type(self).from_pylibcudf(
                plc.datetime.days_in_month(self.to_pylibcudf(mode="read"))
            )

    @property
    def day_of_week(self) -> ColumnBase:
        raise NotImplementedError("day_of_week is currently not implemented.")

    @property
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

    @property
    def values(self):
        """
        Return a CuPy representation of the DateTimeColumn.
        """
        raise NotImplementedError(
            "DateTime Arrays is not yet implemented in cudf"
        )

    def element_indexing(self, index: int):
        result = super().element_indexing(index)
        if cudf.get_option("mode.pandas_compatible"):
            return pd.Timestamp(result)
        return result

    def _get_dt_field(
        self, field: plc.datetime.DatetimeComponent
    ) -> ColumnBase:
        with acquire_spill_lock():
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
            field: self.strftime(format=directive).astype("uint32")
            for field, directive in zip(
                ["year", "week", "day"], ["%G", "%V", "%u"]
            )
        }

    def normalize_binop_value(self, other: DatetimeLikeScalar) -> ScalarLike:
        if isinstance(other, (cudf.Scalar, ColumnBase, cudf.DateOffset)):
            return other

        tz_error_msg = (
            "Cannot perform binary operation on timezone-naive columns"
            " and timezone-aware timestamps."
        )
        if isinstance(other, pd.Timestamp):
            if other.tz is not None:
                raise NotImplementedError(tz_error_msg)
            other = other.to_datetime64()
        elif isinstance(other, pd.Timedelta):
            other = other.to_timedelta64()
        elif isinstance(other, datetime.datetime):
            if other.tzinfo is not None:
                raise NotImplementedError(tz_error_msg)
            other = np.datetime64(other)
        elif isinstance(other, datetime.timedelta):
            other = np.timedelta64(other)

        if isinstance(other, np.datetime64):
            if np.isnat(other):
                other_time_unit = cudf.utils.dtypes.get_time_unit(other)
                if other_time_unit not in {"s", "ms", "ns", "us"}:
                    other_time_unit = "ns"

                return cudf.Scalar(
                    None, dtype=f"datetime64[{other_time_unit}]"
                )

            other = other.astype(self.dtype)
            return cudf.Scalar(other)
        elif isinstance(other, np.timedelta64):
            other_time_unit = cudf.utils.dtypes.get_time_unit(other)

            if np.isnat(other):
                return cudf.Scalar(
                    None,
                    dtype="timedelta64[ns]"
                    if other_time_unit not in {"s", "ms", "ns", "us"}
                    else other.dtype,
                )

            if other_time_unit not in {"s", "ms", "ns", "us"}:
                other = other.astype("timedelta64[s]")

            return cudf.Scalar(other)
        elif isinstance(other, str):
            try:
                return cudf.Scalar(other, dtype=self.dtype)
            except ValueError:
                pass

        return NotImplemented

    def as_datetime_column(self, dtype: Dtype) -> DatetimeColumn:
        if dtype == self.dtype:
            return self
        elif isinstance(dtype, pd.DatetimeTZDtype):
            raise TypeError(
                "Cannot use .astype to convert from timezone-naive dtype to timezone-aware dtype. "
                "Use tz_localize instead."
            )
        return unary.cast(self, dtype=dtype)  # type: ignore[return-value]

    def as_timedelta_column(self, dtype: Dtype) -> None:  # type: ignore[override]
        raise TypeError(
            f"cannot astype a datetimelike from {self.dtype} to {dtype}"
        )

    def as_numerical_column(
        self, dtype: Dtype
    ) -> cudf.core.column.NumericalColumn:
        col = cudf.core.column.NumericalColumn(
            data=self.base_data,  # type: ignore[arg-type]
            dtype=np.dtype(np.int64),
            mask=self.base_mask,
            offset=self.offset,
            size=self.size,
        )
        return cast(cudf.core.column.NumericalColumn, col.astype(dtype))

    def strftime(self, format: str) -> cudf.core.column.StringColumn:
        if len(self) == 0:
            return cast(
                cudf.core.column.StringColumn,
                column.column_empty(0, dtype="object", masked=False),
            )
        if format in _DATETIME_SPECIAL_FORMATS:
            names = as_column(_DATETIME_NAMES)
        else:
            names = cudf.core.column.column_empty(
                0, dtype="object", masked=False
            )
        return string._datetime_to_str_typecast_functions[self.dtype](
            self, format, names
        )

    def as_string_column(self) -> cudf.core.column.StringColumn:
        format = _dtype_to_format_conversion.get(
            self.dtype.name, "%Y-%m-%d %H:%M:%S"
        )
        if cudf.get_option("mode.pandas_compatible"):
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

    def mean(self, skipna=None, min_count: int = 0) -> ScalarLike:
        return pd.Timestamp(
            cast(
                "cudf.core.column.NumericalColumn", self.astype("int64")
            ).mean(skipna=skipna, min_count=min_count),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def std(
        self,
        skipna: bool | None = None,
        min_count: int = 0,
        ddof: int = 1,
    ) -> pd.Timedelta:
        return pd.Timedelta(
            cast("cudf.core.column.NumericalColumn", self.astype("int64")).std(
                skipna=skipna, min_count=min_count, ddof=ddof
            )
            * _unit_to_nanoseconds_conversion[self.time_unit],
        ).as_unit(self.time_unit)

    def median(self, skipna: bool | None = None) -> pd.Timestamp:
        return pd.Timestamp(
            cast(
                "cudf.core.column.NumericalColumn", self.astype("int64")
            ).median(skipna=skipna),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def cov(self, other: DatetimeColumn) -> float:
        if not isinstance(other, DatetimeColumn):
            raise TypeError(
                f"cannot perform cov with types {self.dtype}, {other.dtype}"
            )
        return cast(
            "cudf.core.column.NumericalColumn", self.astype("int64")
        ).cov(cast("cudf.core.column.NumericalColumn", other.astype("int64")))

    def corr(self, other: DatetimeColumn) -> float:
        if not isinstance(other, DatetimeColumn):
            raise TypeError(
                f"cannot perform corr with types {self.dtype}, {other.dtype}"
            )
        return cast(
            "cudf.core.column.NumericalColumn", self.astype("int64")
        ).corr(cast("cudf.core.column.NumericalColumn", other.astype("int64")))

    def quantile(
        self,
        q: np.ndarray,
        interpolation: str,
        exact: bool,
        return_scalar: bool,
    ) -> ColumnBase:
        result = self.astype("int64").quantile(
            q=q,
            interpolation=interpolation,
            exact=exact,
            return_scalar=return_scalar,
        )
        if return_scalar:
            return pd.Timestamp(result, unit=self.time_unit).as_unit(
                self.time_unit
            )
        return result.astype(self.dtype)

    def find_and_replace(
        self,
        to_replace: ColumnBase,
        replacement: ColumnBase,
        all_nan: bool = False,
    ) -> DatetimeColumn:
        return cast(
            DatetimeColumn,
            _datetime_timedelta_find_and_replace(
                original_column=self,
                to_replace=to_replace,
                replacement=replacement,
                all_nan=all_nan,
            ),
        )

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)
        other = self._wrap_binop_normalization(other)
        if other is NotImplemented:
            return NotImplemented
        if isinstance(other, cudf.DateOffset):
            return other._datetime_binop(self, op, reflect=reflect)

        # We check this on `other` before reflection since we already know the
        # dtype of `self`.
        other_is_timedelta = other.dtype.kind == "m"
        other_is_datetime64 = other.dtype.kind == "M"
        lhs, rhs = (other, self) if reflect else (self, other)
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
            out_dtype = cudf.dtype(np.bool_)
        elif op == "__add__" and other_is_timedelta:
            # The only thing we can add to a datetime is a timedelta. This
            # operation is symmetric, i.e. we allow `datetime + timedelta` or
            # `timedelta + datetime`. Both result in DatetimeColumns.
            out_dtype = _resolve_mixed_dtypes(lhs, rhs, "datetime64")
        elif op == "__sub__":
            # Subtracting a datetime from a datetime results in a timedelta.
            if other_is_datetime64:
                out_dtype = _resolve_mixed_dtypes(lhs, rhs, "timedelta64")
            # We can subtract a timedelta from a datetime, but not vice versa.
            # Not only is subtraction antisymmetric (as is normal), it is only
            # well-defined if this operation was not invoked via reflection.
            elif other_is_timedelta and not reflect:
                out_dtype = _resolve_mixed_dtypes(lhs, rhs, "datetime64")
        elif op in {
            "__eq__",
            "__ne__",
            "NULL_EQUALS",
            "NULL_NOT_EQUALS",
        }:
            out_dtype = cudf.dtype(np.bool_)
            if isinstance(other, ColumnBase) and not isinstance(
                other, DatetimeColumn
            ):
                fill_value = op in ("__ne__", "NULL_NOT_EQUALS")
                result = _all_bools_with_nulls(
                    self, other, bool_fill_value=fill_value
                )
                if cudf.get_option("mode.pandas_compatible"):
                    result = result.fillna(fill_value)
                return result

        if out_dtype is None:
            return NotImplemented

        result_col = libcudf.binaryop.binaryop(lhs, rhs, op, out_dtype)
        if out_dtype != cudf.dtype(np.bool_) and op == "__add__":
            return result_col
        elif cudf.get_option(
            "mode.pandas_compatible"
        ) and out_dtype == cudf.dtype(np.bool_):
            return result_col.fillna(op == "__ne__")
        else:
            return result_col

    def indices_of(
        self, value: ScalarLike
    ) -> cudf.core.column.NumericalColumn:
        value = (
            pd.to_datetime(value).to_numpy().astype(self.dtype).astype("int64")
        )
        return self.astype("int64").indices_of(value)

    @property
    def is_unique(self) -> bool:
        return self.astype("int64").is_unique

    def isin(self, values: Sequence) -> ColumnBase:
        return cudf.core.tools.datetimes._isin_datetimelike(self, values)

    def can_cast_safely(self, to_dtype: Dtype) -> bool:
        if to_dtype.kind == "M":  # type: ignore[union-attr]
            to_res, _ = np.datetime_data(to_dtype)
            self_res, _ = np.datetime_data(self.dtype)

            max_int = np.iinfo(cudf.dtype("int64")).max

            max_dist = np.timedelta64(
                self.max().astype(cudf.dtype("int64"), copy=False), self_res
            )
            min_dist = np.timedelta64(
                self.min().astype(cudf.dtype("int64"), copy=False), self_res
            )

            self_delta_dtype = np.timedelta64(0, self_res).dtype

            if max_dist <= np.timedelta64(max_int, to_res).astype(
                self_delta_dtype
            ) and min_dist <= np.timedelta64(max_int, to_res).astype(
                self_delta_dtype
            ):
                return True
            else:
                return False
        elif to_dtype == cudf.dtype("int64") or to_dtype == cudf.dtype("O"):
            # can safely cast to representation, or string
            return True
        else:
            return False

    def _with_type_metadata(self, dtype):
        if isinstance(dtype, pd.DatetimeTZDtype):
            return DatetimeTZColumn(
                data=self.base_data,
                dtype=dtype,
                mask=self.base_mask,
                size=self.size,
                offset=self.offset,
                null_count=self.null_count,
            )
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
        offsets = offsets.astype(f"timedelta64[{self.time_unit}]")  # type: ignore[assignment]

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
        with acquire_spill_lock():
            plc_column = plc.labeling.label_bins(
                self.to_pylibcudf(mode="read"),
                ambiguous_begin.to_pylibcudf(mode="read"),
                plc.labeling.Inclusive.YES,
                ambiguous_end.to_pylibcudf(mode="read"),
                plc.labeling.Inclusive.NO,
            )
            ambiguous = libcudf.column.Column.from_pylibcudf(plc_column)
        ambiguous = ambiguous.notnull()

        # At the start of a non-existent time period, Clock 2 reads less
        # than Clock 1 (which has been turned forward):
        cond = clock_1 > clock_2
        nonexistent_begin = clock_2.apply_boolean_mask(cond)

        # The end of the non-existent time period is what Clock 1 reads
        # at the moment of transition:
        nonexistent_end = clock_1.apply_boolean_mask(cond)
        with acquire_spill_lock():
            plc_column = plc.labeling.label_bins(
                self.to_pylibcudf(mode="read"),
                nonexistent_begin.to_pylibcudf(mode="read"),
                plc.labeling.Inclusive.YES,
                nonexistent_end.to_pylibcudf(mode="read"),
                plc.labeling.Inclusive.NO,
            )
            nonexistent = libcudf.column.Column.from_pylibcudf(plc_column)
        nonexistent = nonexistent.notnull()

        return ambiguous, nonexistent

    def tz_localize(
        self,
        tz: str | None,
        ambiguous: Literal["NaT"] = "NaT",
        nonexistent: Literal["NaT"] = "NaT",
    ):
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
            cudf.Scalar(cudf.NaT, dtype=self.dtype),
        )

        transition_times, offsets = get_tz_data(tzname)
        transition_times_local = (transition_times + offsets).astype(
            localized.dtype
        )
        indices = (
            search_sorted([transition_times_local], [localized], "right") - 1
        )
        offsets_to_utc = offsets.take(indices, nullify=True)
        gmt_data = localized - offsets_to_utc
        return DatetimeTZColumn(
            data=gmt_data.base_data,
            dtype=dtype,
            mask=localized.base_mask,
            size=gmt_data.size,
            offset=gmt_data.offset,
        )

    def tz_convert(self, tz: str | None):
        raise TypeError(
            "Cannot convert tz-naive timestamps, use tz_localize to localize"
        )


class DatetimeTZColumn(DatetimeColumn):
    def __init__(
        self,
        data: Buffer,
        size: int | None,
        dtype: pd.DatetimeTZDtype,
        mask: Buffer | None = None,
        offset: int = 0,
        null_count: int | None = None,
        children: tuple = (),
    ):
        super().__init__(
            data=data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

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
        if arrow_type or nullable:
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        else:
            return self._local_time.to_pandas().tz_localize(
                self.dtype.tz, ambiguous="NaT", nonexistent="NaT"
            )

    def to_arrow(self):
        return pa.compute.assume_timezone(
            self._local_time.to_arrow(), str(self.dtype.tz)
        )

    @functools.cached_property
    def time_unit(self) -> str:
        return self.dtype.unit

    @property
    def _utc_time(self):
        """Return UTC time as naive timestamps."""
        return DatetimeColumn(
            data=self.base_data,
            dtype=_get_base_dtype(self.dtype),
            mask=self.base_mask,
            size=self.size,
            offset=self.offset,
            null_count=self.null_count,
        )

    @property
    def _local_time(self):
        """Return the local time as naive timestamps."""
        transition_times, offsets = get_tz_data(str(self.dtype.tz))
        transition_times = transition_times.astype(_get_base_dtype(self.dtype))
        indices = search_sorted([transition_times], [self], "right") - 1
        offsets_from_utc = offsets.take(indices, nullify=True)
        return self + offsets_from_utc

    def strftime(self, format: str) -> cudf.core.column.StringColumn:
        return self._local_time.strftime(format)

    def as_string_column(self) -> cudf.core.column.StringColumn:
        return self._local_time.as_string_column()

    def as_datetime_column(self, dtype: Dtype) -> DatetimeColumn:
        if isinstance(dtype, pd.DatetimeTZDtype) and dtype != self.dtype:
            if dtype.unit != self.time_unit:
                # TODO: Doesn't check that new unit is valid.
                casted = self._with_type_metadata(dtype)
            else:
                casted = self
            return casted.tz_convert(str(dtype.tz))
        return super().as_datetime_column(dtype)

    def _get_dt_field(
        self, field: plc.datetime.DatetimeComponent
    ) -> ColumnBase:
        with acquire_spill_lock():
            return type(self).from_pylibcudf(
                plc.datetime.extract_datetime_component(
                    self._local_time.to_pylibcudf(mode="read"),
                    field,
                )
            )

    def __repr__(self):
        # Arrow prints the UTC timestamps, but we want to print the
        # local timestamps:
        arr = self._local_time.to_arrow().cast(
            pa.timestamp(self.dtype.unit, str(self.dtype.tz))
        )
        return (
            f"{object.__repr__(self)}\n"
            f"{arr.to_string()}\n"
            f"dtype: {self.dtype}"
        )

    def tz_localize(self, tz: str | None, ambiguous="NaT", nonexistent="NaT"):
        if tz is None:
            return self._local_time
        ambiguous, nonexistent = check_ambiguous_and_nonexistent(
            ambiguous, nonexistent
        )
        raise ValueError(
            "Already localized. "
            "Use `tz_convert` to convert between time zones."
        )

    def tz_convert(self, tz: str | None):
        if tz is None:
            return self._utc_time
        elif tz == str(self.dtype.tz):
            return self.copy()
        utc_time = self._utc_time
        return type(self)(
            data=utc_time.base_data,
            dtype=pd.DatetimeTZDtype(self.time_unit, tz),
            mask=utc_time.base_mask,
            size=utc_time.size,
            offset=utc_time.offset,
        )
