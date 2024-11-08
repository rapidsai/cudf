# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, uint8_t
from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar


cdef extern from "cudf/datetime.hpp" namespace "cudf::datetime" nogil:
    cpdef enum class datetime_component(uint8_t):
        YEAR
        MONTH
        DAY
        WEEKDAY
        HOUR
        MINUTE
        SECOND
        MILLISECOND
        MICROSECOND
        NANOSECOND

    cdef unique_ptr[column] extract_year(const column_view& column) except +
    cdef unique_ptr[column] extract_month(const column_view& column) except +
    cdef unique_ptr[column] extract_day(const column_view& column) except +
    cdef unique_ptr[column] extract_weekday(const column_view& column) except +
    cdef unique_ptr[column] extract_hour(const column_view& column) except +
    cdef unique_ptr[column] extract_minute(const column_view& column) except +
    cdef unique_ptr[column] extract_second(const column_view& column) except +
    cdef unique_ptr[column] extract_millisecond_fraction(
        const column_view& column
    ) except +
    cdef unique_ptr[column] extract_microsecond_fraction(
        const column_view& column
    ) except +
    cdef unique_ptr[column] extract_nanosecond_fraction(
        const column_view& column
    ) except +
    cdef unique_ptr[column] extract_datetime_component(
        const column_view& column,
        datetime_component component
    ) except +

    cpdef enum class rounding_frequency(int32_t):
        DAY
        HOUR
        MINUTE
        SECOND
        MILLISECOND
        MICROSECOND
        NANOSECOND

    cdef unique_ptr[column] ceil_datetimes(
        const column_view& column, rounding_frequency freq
    ) except +
    cdef unique_ptr[column] floor_datetimes(
        const column_view& column, rounding_frequency freq
    ) except +
    cdef unique_ptr[column] round_datetimes(
        const column_view& column, rounding_frequency freq
    ) except +

    cdef unique_ptr[column] add_calendrical_months(
        const column_view& timestamps,
        const column_view& months
    ) except +
    cdef unique_ptr[column] add_calendrical_months(
        const column_view& timestamps,
        const scalar& months
    ) except +
    cdef unique_ptr[column] day_of_year(const column_view& column) except +
    cdef unique_ptr[column] is_leap_year(const column_view& column) except +
    cdef unique_ptr[column] last_day_of_month(
        const column_view& column
    ) except +
    cdef unique_ptr[column] extract_quarter(const column_view& column) except +
    cdef unique_ptr[column] days_in_month(const column_view& column) except +
