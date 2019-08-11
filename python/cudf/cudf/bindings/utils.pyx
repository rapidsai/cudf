from libcpp.vector cimport vector
from libc.stdlib cimport free

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *


cdef cudf_table* table_from_dataframe(df) except? NULL:
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
        col = Column.from_mem_views(data, mask, c_col.null_count)
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
            Column.from_mem_views(data, mask, c_col.null_count)
        )
        free(c_col)
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


def mask_from_devary(py_col):

    cdef gdf_column* c_col = column_view_from_column(py_col)

    cdef pair[bit_mask_t_ptr, gdf_size_type] result

    with nogil:
        result = nans_to_nulls(c_col[0])

    mask = None
    if result.first:
        mask_ptr = int(<uintptr_t>result.first)
        mask = rmm.device_array_from_ptr(
            mask_ptr,
            nelem=calc_chunk_size(len(py_col), mask_bitsize),
            dtype=mask_dtype,
            finalizer=rmm._make_finalizer(mask_ptr, 0)
        )

    return mask
