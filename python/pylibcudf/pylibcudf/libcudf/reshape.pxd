# Copyright (c) 2019-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/reshape.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] interleave_columns(
        table_view source_table
    ) except +libcudf_exception_handler
    cdef unique_ptr[table] tile(
        table_view source_table, size_type count
    ) except +libcudf_exception_handler
