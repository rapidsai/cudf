# Copyright (c) 2024-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.datetime cimport (
    add_calendrical_months as cpp_add_calendrical_months,
    ceil_datetimes as cpp_ceil_datetimes,
    datetime_component,
    day_of_year as cpp_day_of_year,
    days_in_month as cpp_days_in_month,
    extract_datetime_component as cpp_extract_datetime_component,
    extract_quarter as cpp_extract_quarter,
    floor_datetimes as cpp_floor_datetimes,
    is_leap_year as cpp_is_leap_year,
    last_day_of_month as cpp_last_day_of_month,
    round_datetimes as cpp_round_datetimes,
    rounding_frequency,
)

from pylibcudf.libcudf.datetime import \
    datetime_component as DatetimeComponent  # no-cython-lint
from pylibcudf.libcudf.datetime import \
    rounding_frequency as RoundingFrequency  # no-cython-lint

from cython.operator cimport dereference
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .scalar cimport Scalar
from .utils cimport _get_stream

__all__ = [
    "DatetimeComponent",
    "RoundingFrequency",
    "add_calendrical_months",
    "ceil_datetimes",
    "day_of_year",
    "days_in_month",
    "extract_datetime_component",
    "extract_quarter",
    "floor_datetimes",
    "is_leap_year",
    "last_day_of_month",
    "round_datetimes",
]

cpdef Column extract_datetime_component(
    Column input,
    datetime_component component,
    Stream stream=None
):
    """
    Extract a datetime component from a datetime column.

    For details, see :cpp:func:`cudf::extract_datetime_component`.

    Parameters
    ----------
    input : Column
        The column to extract the component from.
    component : DatetimeComponent
        The datetime component to extract.

    Returns
    -------
    Column
        Column with the extracted component.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_extract_datetime_component(input.view(), component, stream.view())
    return Column.from_libcudf(move(result), stream)

cpdef Column ceil_datetimes(
    Column input,
    rounding_frequency freq,
    Stream stream=None
):
    """
    Round datetimes up to the nearest multiple of the given frequency.

    For details, see :cpp:func:`ceil_datetimes`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    freq : rounding_frequency
        The frequency to round up to.

    Returns
    -------
    Column
        Column of the same datetime resolution as the input column.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_ceil_datetimes(input.view(), freq, stream.view())
    return Column.from_libcudf(move(result), stream)

cpdef Column floor_datetimes(
    Column input,
    rounding_frequency freq,
    Stream stream=None
):
    """
    Round datetimes down to the nearest multiple of the given frequency.

    For details, see :cpp:func:`floor_datetimes`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    freq : rounding_frequency
        The frequency to round down to.

    Returns
    -------
    Column
        Column of the same datetime resolution as the input column.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_floor_datetimes(input.view(), freq, stream.view())
    return Column.from_libcudf(move(result), stream)

cpdef Column round_datetimes(
    Column input,
    rounding_frequency freq,
    Stream stream=None
):
    """
    Round datetimes to the nearest multiple of the given frequency.

    For details, see :cpp:func:`round_datetimes`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    freq : rounding_frequency
        The frequency to round to.

    Returns
    -------
    Column
        Column of the same datetime resolution as the input column.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_round_datetimes(input.view(), freq, stream.view())
    return Column.from_libcudf(move(result), stream)

cpdef Column add_calendrical_months(
    Column input,
    ColumnOrScalar months,
    Stream stream=None
):
    """
    Adds or subtracts a number of months from the datetime
    type and returns a timestamp column that is of the same
    type as the input timestamps column.

    For details, see :cpp:func:`add_calendrical_months`.

    Parameters
    ----------
    input : Column
        The column of input timestamp values.
    months : ColumnOrScalar
        The number of months to add.

    Returns
    -------
    Column
        Column of computed timestamps.
    """
    if not isinstance(months, (Column, Scalar)):
        raise TypeError("Must pass a Column or Scalar")

    cdef unique_ptr[column] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_add_calendrical_months(
            input.view(),
            months.view() if ColumnOrScalar is Column else
            dereference(months.get()),
            stream.view()
        )
    return Column.from_libcudf(move(result), stream)

cpdef Column day_of_year(Column input, Stream stream=None):
    """
    Computes the day number since the start of
    the year from the datetime. The value is between
    [1, {365-366}].

    For details, see :cpp:func:`day_of_year`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.

    Returns
    -------
    Column
        Column of day numbers.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_day_of_year(input.view(), stream.view())
    return Column.from_libcudf(move(result), stream)

cpdef Column is_leap_year(Column input, Stream stream=None):
    """
    Check if the year of the given date is a leap year.

    For details, see :cpp:func:`is_leap_year`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.

    Returns
    -------
    Column
        Column of bools indicating whether the given year
        is a leap year.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_is_leap_year(input.view(), stream.view())
    return Column.from_libcudf(move(result), stream)

cpdef Column last_day_of_month(Column input, Stream stream=None):
    """
    Computes the last day of the month.

    For details, see :cpp:func:`last_day_of_month`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.

    Returns
    -------
    Column
        Column of ``TIMESTAMP_DAYS`` representing the last day
        of the month.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_last_day_of_month(input.view(), stream.view())
    return Column.from_libcudf(move(result), stream)

cpdef Column extract_quarter(Column input, Stream stream=None):
    """
    Returns the quarter (ie. a value from {1, 2, 3, 4})
    that the date is in.

    For details, see :cpp:func:`extract_quarter`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.

    Returns
    -------
    Column
        Column indicating which quarter the date is in.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_extract_quarter(input.view(), stream.view())
    return Column.from_libcudf(move(result), stream)

cpdef Column days_in_month(Column input, Stream stream=None):
    """
    Extract the number of days in the month.

    For details, see :cpp:func:`days_in_month`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.

    Returns
    -------
    Column
        Column of the number of days in the given month.
    """
    cdef unique_ptr[column] result

    stream = _get_stream(stream)

    with nogil:
        result = cpp_days_in_month(input.view(), stream.view())
    return Column.from_libcudf(move(result), stream)

DatetimeComponent.__str__ = DatetimeComponent.__repr__
RoundingFrequency.__str__ = RoundingFrequency.__repr__
