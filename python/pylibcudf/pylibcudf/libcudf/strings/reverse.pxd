# Copyright (c) 2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

cdef extern from "cudf/strings/reverse.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] reverse(
        column_view source_strings
    ) except +libcudf_exception_handler
