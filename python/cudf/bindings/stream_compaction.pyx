# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.types cimport table as cudf_table
from cudf.bindings.types import *
from cudf.utils import cudautils
from cudf.dataframe import columnops
from cudf.dataframe.buffer import Buffer
from cudf.dataframe.numerical import NumericalColumn

from libc.stdlib cimport calloc, malloc, free
from libc.stdint cimport uintptr_t
from cudf.bindings.copying cimport cols_view_from_cols, free_table, gather
from cudf.bindings.copying import clone_columns_with_size
from cudf.dataframe.column import Column

def apply_drop_duplicates(in_index, in_cols, subset=None, keep='first'):
    """
    get unique entries of subset columns from input columns

    in_index: index column of input dataframe
    in_cols: list of input columns to filter
    subset:  list of columns to consider for identifying duplicate rows
    keep: keep 'first' entry or 'last' if duplicate rows are found

    out_cols: columns containing only unique rows
    out_index: index of unique rows as column
    """
    cdef col_count=len(in_cols)
    cdef row_count=len(in_index[0])
    if col_count == 0 or row_count == 0:
        return clone_columns_with_size(in_cols, row_count), Column.from_mem_views(rmm.device_array(0, dtype=in_index[0].dtype))
    cdef gdf_column** c_in_cols = cols_view_from_cols(in_cols+in_index)
    cdef cudf_table* c_in_table = new cudf_table(c_in_cols, col_count+1)

    cdef bool keep_first = (keep=='first')

    # check subset == in_cols and subset=None cases
    cdef gdf_column** key_cols
    cdef cudf_table* key_table
    if subset == in_cols or subset is None:
        key_cols = cols_view_from_cols(in_cols)
        key_table = new cudf_table(key_cols, len(in_cols))
    else:
        key_cols = cols_view_from_cols(subset)
        key_table = new cudf_table(key_cols, len(subset))

    cdef cudf_table out_table
    with nogil:
        out_table = drop_duplicates(c_in_table[0], key_table[0], keep_first)

    free_table(key_table, key_cols)
    free_table(c_in_table, c_in_cols)

    #convert table to columns, index
    out_cols = [Column.from_mem_views(*gdf_column_to_column_mem(i)) for i in out_table]
    return (out_cols[:-1], out_cols[-1])

