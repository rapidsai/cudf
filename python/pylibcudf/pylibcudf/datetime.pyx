# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.datetime cimport (
    datetime_component,
    extract_datetime_component as cpp_extract_datetime_component,
    extract_day as cpp_extract_day,
    extract_hour as cpp_extract_hour,
    extract_microsecond_fraction as cpp_extract_microsecond_fraction,
    extract_millisecond_fraction as cpp_extract_millisecond_fraction,
    extract_minute as cpp_extract_minute,
    extract_month as cpp_extract_month,
    extract_nanosecond_fraction as cpp_extract_nanosecond_fraction,
    extract_second as cpp_extract_second,
    extract_weekday as cpp_extract_weekday,
    extract_year as cpp_extract_year,
)

from pylibcudf.libcudf.datetime import \
    datetime_component as DatetimeComponent  # no-cython-lint

from .column cimport Column


cpdef Column extract_year(
    Column values
):
    """
    Extract the year from a datetime column.

    For details, see :cpp:func:`extract_year`.

    Parameters
    ----------
    values : Column
        The column to extract the year from.

    Returns
    -------
    Column
        Column with the extracted years.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_year(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_month(
    Column values
):
    """
    Extract the month from a datetime column.

    For details, see :cpp:func:`extract_month`.

    Parameters
    ----------
    values : Column
        The column to extract the month from.

    Returns
    -------
    Column
        Column with the extracted months.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_month(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_day(
    Column values
):
    """
    Extract the day from a datetime column.

    For details, see :cpp:func:`extract_day`.

    Parameters
    ----------
    values : Column
        The column to extract the day from.

    Returns
    -------
    Column
        Column with the extracted days.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_day(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_weekday(
    Column values
):
    """
    Extract the weekday from a datetime column.

    For details, see :cpp:func:`extract_weekday`.

    Parameters
    ----------
    values : Column
        The column to extract the weekday from.

    Returns
    -------
    Column
        Column with the extracted weekdays.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_weekday(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_hour(
    Column values
):
    """
    Extract the hour from a datetime column.

    For details, see :cpp:func:`extract_hour`.

    Parameters
    ----------
    values : Column
        The column to extract the hour from.

    Returns
    -------
    Column
        Column with the extracted hours.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_hour(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_minute(
    Column values
):
    """
    Extract the minute from a datetime column.

    For details, see :cpp:func:`extract_minute`.

    Parameters
    ----------
    values : Column
        The column to extract the minute from.

    Returns
    -------
    Column
        Column with the extracted minutes.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_minute(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_second(
    Column values
):
    """
    Extract the second from a datetime column.

    For details, see :cpp:func:`extract_second`.

    Parameters
    ----------
    values : Column
        The column to extract the second from.

    Returns
    -------
    Column
        Column with the extracted seconds.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_second(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_millisecond_fraction(
    Column values
):
    """
    Extract the millisecond from a datetime column.

    For details, see :cpp:func:`extract_millisecond_fraction`.

    Parameters
    ----------
    values : Column
        The column to extract the millisecond from.

    Returns
    -------
    Column
        Column with the extracted milliseconds.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_millisecond_fraction(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_microsecond_fraction(
    Column values
):
    """
    Extract the microsecond fraction from a datetime column.

    For details, see :cpp:func:`extract_microsecond_fraction`.

    Parameters
    ----------
    values : Column
        The column to extract the microsecond fraction from.

    Returns
    -------
    Column
        Column with the extracted microsecond fractions.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_microsecond_fraction(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_nanosecond_fraction(
    Column values
):
    """
    Extract the nanosecond fraction from a datetime column.

    For details, see :cpp:func:`extract_nanosecond_fraction`.

    Parameters
    ----------
    values : Column
        The column to extract the nanosecond fraction from.

    Returns
    -------
    Column
        Column with the extracted nanosecond fractions.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_nanosecond_fraction(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_datetime_component(
    Column values,
    datetime_component component
):
    """
    Extract a datetime component from a datetime column.

    For details, see :cpp:func:`cudf::extract_datetime_component`.

    Parameters
    ----------
    values : Column
        The column to extract the component from.
    component : DatetimeComponent
        The datetime component to extract.

    Returns
    -------
    Column
        Column with the extracted component.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_datetime_component(values.view(), component)
    return Column.from_libcudf(move(result))
