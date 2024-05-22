# Copyright (c) 2019-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/reshape.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] interleave_columns(
        table_view source_table
    ) except +
    cdef unique_ptr[table] tile(
        table_view source_table, size_type count
    ) except +
