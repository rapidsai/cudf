# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

cimport cudf._lib.cpp.datetime as libcudf_datetime
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.filling cimport calendrical_month_sequence
from cudf._lib.cpp.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def add_months(Column col, Column months):
    # months must be int16 dtype
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()
    cdef column_view months_view = months.view()

    with nogil:
        c_result = move(
            libcudf_datetime.add_calendrical_months(
                col_view,
                months_view
            )
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def extract_datetime_component(Column col, object field):

    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()

    with nogil:
        if field == "year":
            c_result = move(libcudf_datetime.extract_year(col_view))
        elif field == "month":
            c_result = move(libcudf_datetime.extract_month(col_view))
        elif field == "day":
            c_result = move(libcudf_datetime.extract_day(col_view))
        elif field == "weekday":
            c_result = move(libcudf_datetime.extract_weekday(col_view))
        elif field == "hour":
            c_result = move(libcudf_datetime.extract_hour(col_view))
        elif field == "minute":
            c_result = move(libcudf_datetime.extract_minute(col_view))
        elif field == "second":
            c_result = move(libcudf_datetime.extract_second(col_view))
        elif field == "millisecond":
            c_result = move(
                libcudf_datetime.extract_millisecond_fraction(col_view)
            )
        elif field == "microsecond":
            c_result = move(
                libcudf_datetime.extract_microsecond_fraction(col_view)
            )
        elif field == "nanosecond":
            c_result = move(
                libcudf_datetime.extract_nanosecond_fraction(col_view)
            )
        elif field == "day_of_year":
            c_result = move(libcudf_datetime.day_of_year(col_view))
        else:
            raise ValueError(f"Invalid datetime field: '{field}'")

    result = Column.from_unique_ptr(move(c_result))

    if field == "weekday":
        # Pandas counts Monday-Sunday as 0-6
        # while libcudf counts Monday-Sunday as 1-7
        result = result - result.dtype.type(1)

    return result


cdef libcudf_datetime.rounding_frequency _get_rounding_frequency(object freq):
    cdef libcudf_datetime.rounding_frequency freq_val

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.resolution_string.html
    if freq == "D":
        freq_val = libcudf_datetime.rounding_frequency.DAY
    elif freq == "H":
        freq_val = libcudf_datetime.rounding_frequency.HOUR
    elif freq in ("T", "min"):
        freq_val = libcudf_datetime.rounding_frequency.MINUTE
    elif freq == "S":
        freq_val = libcudf_datetime.rounding_frequency.SECOND
    elif freq in ("L", "ms"):
        freq_val = libcudf_datetime.rounding_frequency.MILLISECOND
    elif freq in ("U", "us"):
        freq_val = libcudf_datetime.rounding_frequency.MICROSECOND
    elif freq == "N":
        freq_val = libcudf_datetime.rounding_frequency.NANOSECOND
    else:
        raise ValueError(f"Invalid resolution: '{freq}'")
    return freq_val


@acquire_spill_lock()
def ceil_datetime(Column col, object freq):
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()
    cdef libcudf_datetime.rounding_frequency freq_val = \
        _get_rounding_frequency(freq)

    with nogil:
        c_result = move(libcudf_datetime.ceil_datetimes(col_view, freq_val))

    result = Column.from_unique_ptr(move(c_result))
    return result


@acquire_spill_lock()
def floor_datetime(Column col, object freq):
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()
    cdef libcudf_datetime.rounding_frequency freq_val = \
        _get_rounding_frequency(freq)

    with nogil:
        c_result = move(libcudf_datetime.floor_datetimes(col_view, freq_val))

    result = Column.from_unique_ptr(move(c_result))
    return result


@acquire_spill_lock()
def round_datetime(Column col, object freq):
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()
    cdef libcudf_datetime.rounding_frequency freq_val = \
        _get_rounding_frequency(freq)

    with nogil:
        c_result = move(libcudf_datetime.round_datetimes(col_view, freq_val))

    result = Column.from_unique_ptr(move(c_result))
    return result


@acquire_spill_lock()
def is_leap_year(Column col):
    """Returns a boolean indicator whether the year of the date is a leap year
    """
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()

    with nogil:
        c_result = move(libcudf_datetime.is_leap_year(col_view))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def date_range(DeviceScalar start, size_type n, offset):
    cdef unique_ptr[column] c_result
    cdef size_type months = (
        offset.kwds.get("years", 0) * 12
        + offset.kwds.get("months", 0)
    )

    with nogil:
        c_result = move(calendrical_month_sequence(
            n,
            start.c_value.get()[0],
            months
        ))
    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def extract_quarter(Column col):
    """
    Returns a column which contains the corresponding quarter of the year
    for every timestamp inside the input column.
    """
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()

    with nogil:
        c_result = move(libcudf_datetime.extract_quarter(col_view))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def days_in_month(Column col):
    """Extracts the number of days in the month of the date
    """
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()

    with nogil:
        c_result = move(libcudf_datetime.days_in_month(col_view))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def last_day_of_month(Column col):
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()

    with nogil:
        c_result = move(libcudf_datetime.last_day_of_month(col_view))

    return Column.from_unique_ptr(move(c_result))
