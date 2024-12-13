# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair

from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.aggregation cimport rolling_aggregation
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/rolling.hpp" namespace "cudf" nogil:
    cpdef enum class window_type(int32_t):
        LEFT_CLOSED
        RIGHT_CLOSED
        CLOSED
        OPEN

    cdef unique_ptr[column] rolling_window(
        column_view source,
        column_view preceding_window,
        column_view following_window,
        size_type min_periods,
        rolling_aggregation& agg) except +libcudf_exception_handler

    cdef unique_ptr[column] rolling_window(
        column_view source,
        size_type preceding_window,
        size_type following_window,
        size_type min_periods,
        rolling_aggregation& agg) except +libcudf_exception_handler

    cdef pair[unique_ptr[column], unique_ptr[column]] windows_from_offset(
        column_view input,
        scalar length,
        scalar offset,
        window_type,
        bool) except +libcudf_exception_handler
