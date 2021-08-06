from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


cdef extern from "cudf/datetime.hpp" namespace "cudf::datetime" nogil:
    cdef unique_ptr[column] extract_year(const column_view& column) except +
    cdef unique_ptr[column] extract_month(const column_view& column) except +
    cdef unique_ptr[column] extract_day(const column_view& column) except +
    cdef unique_ptr[column] extract_weekday(const column_view& column) except +
    cdef unique_ptr[column] extract_hour(const column_view& column) except +
    cdef unique_ptr[column] extract_minute(const column_view& column) except +
    cdef unique_ptr[column] extract_second(const column_view& column) except +
    cdef unique_ptr[column] add_calendrical_months(
        const column_view& timestamps,
        const column_view& months
    ) except +
    cdef unique_ptr[column] day_of_year(const column_view& column) except +
    cdef unique_ptr[column] is_leap_year(const column_view& column) except +
    cdef unique_ptr[column] last_day_of_month(
        const column_view& column
    ) except +
