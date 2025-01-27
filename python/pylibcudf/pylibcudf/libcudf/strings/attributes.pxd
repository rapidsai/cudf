# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "cudf/strings/attributes.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] count_characters(
        column_view source_strings) except +libcudf_exception_handler

    cdef unique_ptr[column] count_bytes(
        column_view source_strings) except +libcudf_exception_handler

    cdef unique_ptr[column] code_points(
        column_view source_strings) except +libcudf_exception_handler
