from libcpp.vector cimport vector
from libc.stdlib cimport free

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *


cdef cudf_table* table_from_dataframe(df):
    cdef cudf_table* c_table
    cdef vector[gdf_column*] c_columns
    for col_name in df:
        c_columns.push_back(column_view_from_column(
            df[col_name]._column))
    c_table = new cudf_table(c_columns)
    return c_table


cdef dataframe_from_table(cudf_table* table, colnames):
    cdef gdf_column* c_col
    from cudf.dataframe.column import Column
    df = cudf.DataFrame()
    for i in range(table[0].num_columns()):
        c_col = table[0].get_column(i)
        data, mask = gdf_column_to_column_mem(c_col)
        col = Column.from_mem_views(data, mask)
        df.add_column(
            name=colnames[i],
            data=col
        )
        free(c_col)
    return df


cdef columns_from_table(cudf_table* table):
    from cudf.dataframe.column import Column
    cdef gdf_column* c_col
    columns = []
    for i in range(table[0].num_columns()):
        c_col = table[0].get_column(i)
        data, mask = gdf_column_to_column_mem(c_col)
        columns.append(
            Column.from_mem_views(data, mask)
        )
        free(c_col)
    return columns


cdef cudf_table* table_from_columns(columns):
    cdef cudf_table* c_table
    cdef vector[gdf_column*] c_columns
    cdef gdf_column* c_col
    for col in columns:
        c_col = column_view_from_column(col)
        c_columns.push_back(c_col)
    c_table = new cudf_table(c_columns)
    return c_table
