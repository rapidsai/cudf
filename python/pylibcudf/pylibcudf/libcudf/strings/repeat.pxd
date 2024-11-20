# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/repeat_strings.hpp" namespace "cudf::strings" \
        nogil:

    cdef unique_ptr[column] repeat_strings(
        column_view input,
        size_type repeat_times) except +libcudf_exception_handler

    cdef unique_ptr[column] repeat_strings(
        column_view input,
        column_view repeat_times) except +libcudf_exception_handler
