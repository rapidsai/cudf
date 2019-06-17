from libcpp.vector cimport vector

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.types cimport *


cdef cudf_table* table_from_dataframe(df):
    cdef cudf_table* c_table
    cdef vector[gdf_column*] c_columns
    for col_name in df:
        c_columns.push_back(column_view_from_column(
            df[col_name]._column))
        c_table = new cudf_table(c_columns)
    return c_table


cdef dataframe_from_table(cudf_table* table, colnames):
    from cudf.dataframe.column import Column
    df = cudf.DataFrame()
    for i in range(table[0].num_columns()):
        data, mask = gdf_column_to_column_mem(table[0].get_column(i))
        df.add_column(
            name=colnames[i],
            data=Column.from_mem_views(data, mask)
        )
    return df
