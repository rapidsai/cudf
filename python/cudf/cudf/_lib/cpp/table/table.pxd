# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table_view cimport mutable_table_view, table_view
from cudf._lib.cpp.types cimport size_type


cdef extern from "cudf/table/table.hpp" namespace "cudf" nogil:
    cdef cppclass table:
        table(const table&) except +
        table(vector[unique_ptr[column]]&& columns) except +
        table(table_view) except +
        size_type num_columns() except +
        table_view view() except +
        mutable_table_view mutable_view() except +
        vector[unique_ptr[column]] release() except +
