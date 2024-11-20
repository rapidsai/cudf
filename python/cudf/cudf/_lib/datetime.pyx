# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import warnings

from cudf.core.buffer import acquire_spill_lock

cimport pylibcudf.libcudf.datetime as libcudf_datetime
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.datetime import DatetimeComponent, RoundingFrequency

from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar
import pylibcudf as plc


@acquire_spill_lock()
def add_months(Column col, Column months):
    # months must be int16 dtype
    return Column.from_pylibcudf(
        plc.datetime.add_calendrical_months(
            col.to_pylibcudf(mode="read"),
            months.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def extract_datetime_component(Column col, object field):
    component_names = {
        "year": DatetimeComponent.YEAR,
        "month": DatetimeComponent.MONTH,
        "day": DatetimeComponent.DAY,
        "weekday": DatetimeComponent.WEEKDAY,
        "hour": DatetimeComponent.HOUR,
        "minute": DatetimeComponent.MINUTE,
        "second": DatetimeComponent.SECOND,
        "millisecond": DatetimeComponent.MILLISECOND,
        "microsecond": DatetimeComponent.MICROSECOND,
        "nanosecond": DatetimeComponent.NANOSECOND,
    }
    if field == "day_of_year":
        result = Column.from_pylibcudf(
            plc.datetime.day_of_year(
                col.to_pylibcudf(mode="read")
            )
        )
    elif field in component_names:
        result = Column.from_pylibcudf(
            plc.datetime.extract_datetime_component(
                col.to_pylibcudf(mode="read"),
                component_names[field],
            )
        )
        if field == "weekday":
            # Pandas counts Monday-Sunday as 0-6
            # while libcudf counts Monday-Sunday as 1-7
            result = result - result.dtype.type(1)
    else:
        raise ValueError(f"Invalid field: '{field}'")

    return result


cdef libcudf_datetime.rounding_frequency _get_rounding_frequency(object freq):
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
            f"FutureWarning: {freq} is deprecated and will be "
            "removed in a future version, please use "
            f"{old_to_new_freq_map[freq]} instead.",
            FutureWarning
        )
        freq = old_to_new_freq_map.get(freq)
    rounding_fequency_map = {
        "D": RoundingFrequency.DAY,
        "h": RoundingFrequency.HOUR,
        "min": RoundingFrequency.MINUTE,
        "s": RoundingFrequency.SECOND,
        "ms": RoundingFrequency.MILLISECOND,
        "us": RoundingFrequency.MICROSECOND,
        "ns": RoundingFrequency.NANOSECOND,
    }
    if freq in rounding_fequency_map:
        return rounding_fequency_map[freq]
    else:
        raise ValueError(f"Invalid resolution: '{freq}'")


@acquire_spill_lock()
def ceil_datetime(Column col, object freq):
    return Column.from_pylibcudf(
        plc.datetime.ceil_datetimes(
            col.to_pylibcudf(mode="read"),
            _get_rounding_frequency(freq),
        )
    )


@acquire_spill_lock()
def floor_datetime(Column col, object freq):
    return Column.from_pylibcudf(
        plc.datetime.floor_datetimes(
            col.to_pylibcudf(mode="read"),
            _get_rounding_frequency(freq),
        )
    )


@acquire_spill_lock()
def round_datetime(Column col, object freq):
    return Column.from_pylibcudf(
        plc.datetime.round_datetimes(
            col.to_pylibcudf(mode="read"),
            _get_rounding_frequency(freq),
        )
    )


@acquire_spill_lock()
def is_leap_year(Column col):
    """Returns a boolean indicator whether the year of the date is a leap year
    """
    return Column.from_pylibcudf(
        plc.datetime.is_leap_year(
            col.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def date_range(DeviceScalar start, size_type n, offset):
    cdef size_type months = (
        offset.kwds.get("years", 0) * 12
        + offset.kwds.get("months", 0)
    )
    return Column.from_pylibcudf(
        plc.filling.calendrical_month_sequence(
            n,
            start.c_value,
            months,
        )
    )


@acquire_spill_lock()
def extract_quarter(Column col):
    """
    Returns a column which contains the corresponding quarter of the year
    for every timestamp inside the input column.
    """
    return Column.from_pylibcudf(
        plc.datetime.extract_quarter(
            col.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def days_in_month(Column col):
    """Extracts the number of days in the month of the date
    """
    return Column.from_pylibcudf(
        plc.datetime.days_in_month(
            col.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def last_day_of_month(Column col):
    return Column.from_pylibcudf(
        plc.datetime.last_day_of_month(
            col.to_pylibcudf(mode="read")
        )
    )
