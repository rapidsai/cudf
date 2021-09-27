# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport mutable_table_view, table_view


cdef table_view table_view_from_columns(columns) except *
cdef table_view table_view_from_table(tbl, ignore_index=*) except*
