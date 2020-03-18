from libcpp.memory cimport unique_ptr

from cudf._libxx.cpp.column cimport column
from cudf._libxx.cpp.column_view cimport column_view
from cudf._libxx.cpp.table cimport table
from cudf._libxx.cpp.table_view cimport table_view


cdef extern from "cudf/datetime.hpp" namespace "cudf::datetime" nogil:
    cdef unique_ptr[column] extract_year(const column_view& column)
    cdef unique_ptr[column] extract_month(const column_view& column)
    cdef unique_ptr[column] extract_day(const column_view& column)
    cdef unique_ptr[column] extract_weekday(const column_view& column)
    cdef unique_ptr[column] extract_hour(const column_view& column)
    cdef unique_ptr[column] extract_minute(const column_view& column)
    cdef unique_ptr[column] extract_second(const column_view& column)
