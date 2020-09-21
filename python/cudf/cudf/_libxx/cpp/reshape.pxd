# Copyright (c) 2019, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view

cdef extern from "cudf/reshape.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[column] interleave_columns(
        table_view source_table
    ) except +
    cdef unique_ptr[table] tile(
        table_view source_table, size_type count
    ) except +
