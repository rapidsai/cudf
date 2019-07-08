# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdlib cimport free
from libcpp.vector cimport vector

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.utils cimport *
from cudf.bindings.copying cimport cols_view_from_cols, free_table
from cudf.bindings.copying import clone_columns_with_size
from cudf.dataframe.column import Column
from cudf.dataframe.buffer import Buffer

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
    elif keep == False:
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
        out_table = drop_duplicates(
            c_in_table[0], key_table[0], keep_first
        )

    free_table(key_table, key_cols)
    free_table(c_in_table, c_in_cols)

    #convert table to columns, index
    out_cols = [Column.from_mem_views(*gdf_column_to_column_mem(i)) for i in out_table]
    return (out_cols[:-1], out_cols[-1])


def apply_boolean_mask_column(col, mask):
    """
    Apply the given boolean mask ``mask`` to the Column ``col``
    """
    from cudf.dataframe.columnops import column_empty_like

    cdef gdf_column *c_col = column_view_from_column(col)
    cdef gdf_column *c_mask = column_view_from_column(mask)
    cdef gdf_column c_result
    with nogil:
        c_result = apply_boolean_mask(c_col[0], c_mask[0])
    free(c_col)
    free(c_mask)
    if c_result.data is NULL:
        return column_empty_like(col, newsize=0)
    data, mask = gdf_column_to_column_mem(&c_result)
    return Column.from_mem_views(data, mask)


def drop_nulls_column(col):
    cdef gdf_column *c_col = column_view_from_column(col)
    cdef gdf_column c_result
    with nogil:
        c_result = drop_nulls(c_col[0])
    free(c_col)
    data, mask = gdf_column_to_column_mem(&c_result)
    return Column.from_mem_views(data, mask)


def apply_boolean_mask_table(cols, mask):
    pass


def drop_nulls_table(cols, how="any", subset=None):
    cdef cudf_table c_out_table
    cdef cudf_table* c_in_table = table_from_columns(cols)

    cdef any_or_all drop_if
    if how == "any":
        drop_if = ANY
    else:
        drop_if = ALL

    cdef vector[gdf_index_type] column_indices
    if subset is not None:
        column_indices = subset
    else:
        column_indices = range(len(cols))

    print(column_indices)
    with nogil:
        c_out_table = drop_nulls(c_in_table[0], column_indices, drop_if)

    free(c_in_table)

    return columns_from_table(&c_out_table)


def apply_apply_boolean_mask(inp, mask):
    if isinstance(inp, Column):
        return apply_boolean_mask_column(inp, mask)
    else:
        return apply_boolean_mask_table(inp, mask)


def apply_drop_nulls(inp, how="any", subset=None):
    if isinstance(inp, Column):
        return drop_nulls_column(inp)
    else:
        return drop_nulls_table(inp, how=how, subset=subset)
