# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdlib cimport free

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.utils cimport *

from cudf.bindings.copying cimport cols_view_from_cols, free_table
from cudf.bindings.copying import clone_columns_with_size
from cudf.dataframe.column import Column
from cudf.bindings.stream_compaction cimport *


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
        return clone_columns_with_size(in_cols, row_count), in_index[0].copy()

    cdef gdf_column** c_in_cols = cols_view_from_cols(in_cols+in_index)
    cdef cudf_table* c_in_table = new cudf_table(c_in_cols, col_count+1)

    cdef duplicate_keep_option keep_first
    if keep == 'first':
        keep_first = duplicate_keep_option.KEEP_FIRST
    elif keep == 'last':
        keep_first = duplicate_keep_option.KEEP_LAST
    elif keep is False:
        keep_first = duplicate_keep_option.KEEP_NONE
    else:
        raise ValueError('keep must be either "first", "last" or False')

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

    # convert table to columns, index
    out_cols = [
        Column.from_mem_views(
            *gdf_column_to_column_mem(i)
        ) for i in out_table
    ]
    return (out_cols[:-1], out_cols[-1])


def apply_apply_boolean_mask(cols, mask):

    cdef cudf_table  c_out_table
    cdef cudf_table* c_in_table = table_from_columns(cols)
    cdef gdf_column* c_mask_col = column_view_from_column(mask)

    with nogil:
        c_out_table = apply_boolean_mask(c_in_table[0], c_mask_col[0])

    free(c_in_table)
    free(c_mask_col)

    return columns_from_table(&c_out_table)


def apply_drop_nulls(cols, how="any", subset=None, thresh=None):
    cdef cudf_table c_out_table
    cdef cudf_table* c_in_table = table_from_columns(cols)
    cdef cudf_table* c_keys_table = (table_from_columns(cols) if subset is None
                                     else table_from_columns(subset))

    cdef any_or_all drop_if
    cdef gdf_size_type keep_threshold = len(cols)

    # If the threshold is speified, use it
    # otherwise set it based on how
    if thresh:
        keep_threshold = thresh
    elif how == "all":
        keep_threshold = 0

    with nogil:
        c_out_table = drop_nulls(c_in_table[0], c_keys_table[0],
                                 keep_threshold)

    free(c_in_table)
    free(c_keys_table)

    return columns_from_table(&c_out_table)
