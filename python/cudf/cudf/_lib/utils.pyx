# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table
from cudf._lib.cpp.column.column cimport column_view
from cudf._lib.cpp.table.table cimport table_view


cdef vector[column_view] make_column_views(object columns):
    cdef vector[column_view] views
    views.reserve(len(columns))
    for col in columns:
        views.push_back((<Column> col).view())
    return views


cdef vector[table_view] make_table_views(object tables):
    cdef vector[table_view] views
    views.reserve(len(tables))
    for tbl in tables:
        views.push_back((<Table> tbl).view())
    return views


cdef vector[table_view] make_table_data_views(object tables):
    cdef vector[table_view] views
    views.reserve(len(tables))
    for tbl in tables:
        views.push_back((<Table> tbl).data_view())
    return views
