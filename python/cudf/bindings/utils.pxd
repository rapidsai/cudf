from cudf.bindings.cudf_cpp cimport cudf_table

cdef cudf_table* table_from_dataframe(df)
cdef dataframe_from_table(cudf_table* table, colnames)
cdef columns_from_table(cudf_table* table)
cdef cudf_table* table_from_columns(columns)
