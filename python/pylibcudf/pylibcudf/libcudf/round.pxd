# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "cudf/round.hpp" namespace "cudf" nogil:

    cpdef enum class rounding_method(int32_t):
        HALF_UP
        HALF_EVEN

    cdef unique_ptr[column] round (
        const column_view& input,
        int32_t decimal_places,
        rounding_method method,
    ) except +libcudf_exception_handler
