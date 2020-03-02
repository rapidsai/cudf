# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.libcpp.functional cimport reference_wrapper
from cudf._libxx.cpp.scalar.scalar cimport scalar
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
from cudf._libxx.cpp.types cimport size_type

cdef extern from "cudf/copying.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] gather (
        table_view source_table,
        column_view gather_map
    ) except +

    cdef unique_ptr[table] shift(
        table_view input,
        size_type offset,
        vector[reference_wrapper[scalar]] fill_values
    ) except +
