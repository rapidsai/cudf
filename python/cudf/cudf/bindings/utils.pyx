# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libcpp.vector cimport vector

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


cdef table_to_dataframe(cudf_table* c_table, int_col_names=False):
    """
    Util to create a Python cudf.DataFrame from a libcudf cudf_table.

    Notes
        This function frees each gdf_column after use.

    Parameters
    ----------
    c_table : cudf_table*
        A pointer to the source cudf_table.
    int_col_names : bool; optional
        A flag indicating string column names should be cast
        to integers after decoding (default: False).
    """
    from cudf.dataframe.dataframe import DataFrame
    cdef i
    cdef gdf_column* c_col
    df = DataFrame()
    for i in range(c_table[0].num_columns()):
        c_col = c_table[0].get_column(i)
        col = gdf_column_to_column(c_col, int_col_names)
        df.add_column(data=col, name=col.name)
        free_column(c_col)
    return df


cdef columns_from_table(cudf_table* c_table, int_col_names=False):
    """
    Util to create a Python list of cudf.Columns from a libcudf cudf_table.

    Notes
        This function frees each gdf_column after use.

    Parameters
    ----------
    c_table : cudf_table*
        A pointer to the source cudf_table.
    int_col_names : bool; optional
        A flag indicating string column names should be cast
        to integers after decoding (default: False).
    """
    columns = []
    cdef i
    cdef gdf_column* c_col
    for i in range(c_table[0].num_columns()):
        c_col = c_table[0].get_column(i)
        col = gdf_column_to_column(c_col, int_col_names)
        columns.append(col)
        free_column(c_col)
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
    from cudf.dataframe.columnops import as_column

    py_col = as_column(py_col)

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
