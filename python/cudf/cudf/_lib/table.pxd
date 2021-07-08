# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport (
    table_view, mutable_table_view
)


cdef class Table:
    cdef dict __dict__

    cdef table_view view(self) except *
    cdef mutable_table_view mutable_view(self) except *
    cdef table_view data_view(self) except *
    cdef mutable_table_view mutable_data_view(self) except *
    cdef table_view index_view(self) except *
    cdef mutable_table_view mutable_index_view(self) except *

cdef table_view make_table_view(columns) except *
cdef mutable_table_view make_mutable_table_view(columns) except *
cdef columns_from_ptr(unique_ptr[table] c_tbl)
