# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.vector cimport vector
from cudf._lib.cpp.column.column cimport column_view
from cudf._lib.cpp.table.table cimport table, table_view
from cudf._lib.table cimport Table

from libcpp.memory cimport unique_ptr

cdef vector[column_view] make_column_views(object columns) except*
cdef vector[table_view] make_table_views(object tables) except*
cdef vector[table_view] make_table_data_views(object tables) except*
cdef vector[string] get_column_names(Table table, object index) except*
cdef data_from_unique_ptr(
    unique_ptr[table] c_tbl, column_names, index_names=*)
cdef data_from_table_view(
    table_view tv, object owner, object column_names, object index_names=*)
