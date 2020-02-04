# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.vector cimport vector

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *

from io import BytesIO, StringIO


cdef cudf_table* table_from_dataframe(df) except? NULL:
    cdef cudf_table* c_table
    cdef vector[gdf_column*] c_columns
    for col_name in df:
        col = df[col_name]._column
        c_columns.push_back(column_view_from_column(col))
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
    from cudf.core.dataframe import DataFrame
    cdef i
    cdef gdf_column* c_col
    df = DataFrame()
    for i in range(c_table[0].num_columns()):
        c_col = c_table[0].get_column(i)
        col = gdf_column_to_column(c_col)
        name = None
        if c_col.col_name is not NULL:
            name = c_col.col_name.decode()
        if int_col_names:
            name = int(name)
        df.insert(i, name, col)
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
        col = gdf_column_to_column(c_col)
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


cdef const unsigned char[::1] view_of_buffer(filepath_or_buffer) except *:
    """
    Util to obtain a 1-D char-typed memoryview into a Python buffer

    Parameters
    ----------
    filepath_or_buffer : filepath or buffer
        The Python object from which to retrieve a memoryview. To succeed, the
        object needs to export the Python buffer protocol interface.
    """
    cdef const unsigned char[::1] buffer = None
    if isinstance(filepath_or_buffer, BytesIO):
        buffer = filepath_or_buffer.getbuffer()
    elif isinstance(filepath_or_buffer, StringIO):
        buffer = filepath_or_buffer.read().encode()
    elif isinstance(filepath_or_buffer, bytes):
        buffer = filepath_or_buffer
    return buffer
