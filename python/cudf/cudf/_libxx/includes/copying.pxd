# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._libxx.includes.table.table cimport table
from cudf._libxx.includes.table.table_view cimport table_view
from cudf._libxx.includes.column.column_view cimport column_view

cdef extern from "cudf/copying.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] gather (
        table_view source_table,
        column_view gather_map
    ) except +
