# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table


cdef extern from "cudf/strings/find_multiple.hpp" namespace "cudf::strings" \
        nogil:

    cdef unique_ptr[table] contains_multiple(
        column_view input,
        column_view targets) except +libcudf_exception_handler

    cdef unique_ptr[column] find_multiple(
        column_view input,
        column_view targets) except +libcudf_exception_handler
