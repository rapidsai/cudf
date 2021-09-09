# Copyright (c) 2019, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.types cimport data_type
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type


cdef extern from "cudf/reshape.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] interleave_columns(
        table_view source_table
    ) except +
    cdef unique_ptr[table] tile(
        table_view source_table, size_type count
    ) except +
    cdef unique_ptr[table] one_hot_encoding(
        column_view input_column,
        column_view categories,
        data_type output_type
    ) except +
    
