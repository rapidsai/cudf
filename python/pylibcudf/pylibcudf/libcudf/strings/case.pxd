# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "cudf/strings/case.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] capitalize(
        const column_view & input) except +

    cdef unique_ptr[column] is_title(
        const column_view & input) except +

    cdef unique_ptr[column] to_lower(
        const column_view & strings) except +

    cdef unique_ptr[column] to_upper(
        const column_view & strings) except +

    cdef unique_ptr[column] swapcase(
        const column_view & strings) except +
