# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar


cdef extern from "cudf/datetime.hpp" namespace "cudf::datetime" nogil:
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

    ctypedef enum rounding_frequency "cudf::datetime::rounding_frequency":
        DAY "cudf::datetime::rounding_frequency::DAY"
        HOUR "cudf::datetime::rounding_frequency::HOUR"
        MINUTE "cudf::datetime::rounding_frequency::MINUTE"
        SECOND "cudf::datetime::rounding_frequency::SECOND"
        MILLISECOND "cudf::datetime::rounding_frequency::MILLISECOND"
        MICROSECOND "cudf::datetime::rounding_frequency::MICROSECOND"
        NANOSECOND "cudf::datetime::rounding_frequency::NANOSECOND"

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
    cdef unique_ptr[column] day_of_year(const column_view& column) except +
    cdef unique_ptr[column] is_leap_year(const column_view& column) except +
    cdef unique_ptr[column] last_day_of_month(
        const column_view& column
    ) except +
    cdef unique_ptr[column] extract_quarter(const column_view& column) except +
    cdef unique_ptr[column] days_in_month(const column_view& column) except +
