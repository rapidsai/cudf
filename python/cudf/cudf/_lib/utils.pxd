# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.column.column cimport column_view
from cudf._lib.pylibcudf.libcudf.table.table cimport table, table_view


cdef data_from_unique_ptr(
    unique_ptr[table] c_tbl, column_names, index_names=*)
cdef data_from_pylibcudf_table(tbl, column_names, index_names=*)
cdef data_from_table_view(
    table_view tv, object owner, object column_names, object index_names=*)
cdef table_view table_view_from_columns(columns) except *
cdef table_view table_view_from_table(tbl, ignore_index=*) except*
cdef columns_from_unique_ptr(unique_ptr[table] c_tbl)
cdef columns_from_table_view(table_view tv, object owners)
cdef columns_from_pylibcudf_table(tbl)
