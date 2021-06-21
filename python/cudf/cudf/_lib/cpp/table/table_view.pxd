# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.vector cimport vector

from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.column.column_view cimport (
    column_view,
    mutable_column_view
)

cdef extern from "cudf/table/table_view.hpp" namespace "cudf" nogil:
    cdef cppclass table_view:
        table_view() except +
        table_view(const vector[column_view]) except +
        column_view column(size_type column_index) except +
        size_type num_columns() except +
        size_type num_rows() except +
        table_view select(vector[size_type] column_indices) except +

    cdef cppclass mutable_table_view:
        mutable_table_view() except +
        mutable_table_view(const vector[mutable_column_view]) except +
        mutable_column_view column(size_type column_index) except +
        size_type num_columns() except +
        size_type num_rows() except +
