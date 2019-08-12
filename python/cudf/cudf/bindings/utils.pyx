# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libcpp.vector cimport vector
from libc.stdlib cimport free

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *


cdef cudf_table* table_from_dataframe(df) except? NULL:
    cdef cudf_table* c_table
    cdef vector[gdf_column*] c_columns
    for col_name in df:
        col = df[col_name]._column
        c_columns.push_back(column_view_from_column(col, col.name))
    c_table = new cudf_table(c_columns)
    return c_table


cdef table_to_dataframe(cudf_table* c_table):
    import cudf
    cdef gdf_column* c_col
    df = cudf.DataFrame()
    for c_col in c_table[0]:
        col = gdf_column_to_column(c_col)
        df.add_column(data=col, name=col.name)
    return df


cdef columns_from_table(cudf_table* c_table):
    columns = []
    cdef gdf_column* c_col
    for c_col in c_table[0]:
        columns.append(gdf_column_to_column(c_col))
    return columns


cdef cudf_table* table_from_columns(columns) except? NULL:
    cdef cudf_table* c_table
    cdef vector[gdf_column*] c_columns
    cdef gdf_column* c_col
    for col in columns:
        c_col = column_view_from_column(col)
        c_columns.push_back(c_col)
    c_table = new cudf_table(c_columns)
    return c_table
