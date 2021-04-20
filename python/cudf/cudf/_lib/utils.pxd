# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.vector cimport vector
from cudf._lib.cpp.column.column cimport column_view
from cudf._lib.cpp.table.table cimport table_view
from cudf._lib.table cimport Table

cdef vector[column_view] make_column_views(object columns) except*
cdef vector[table_view] make_table_views(object tables) except*
cdef vector[table_view] make_table_data_views(object tables) except*
cdef vector[string] get_column_names(Table table, object index) except*
