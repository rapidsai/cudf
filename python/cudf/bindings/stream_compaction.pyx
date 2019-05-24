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

def get_unique_indices(in_index, in_cols, subset=None, keep='first'):
    """
    get unique entries of subset columns from input columns

    in_index: index column of input dataframe
    in_cols: list of input columns to filter
    subset:  list of columns to consider for identifying duplicate rows
    keep: keep 'first' entry or 'last' if duplicate rows are found

    out_index: index of unique rows as column
    """
    cdef col_count=len(in_cols)
    if col_count == 0:
        return clone_columns_with_size(in_index, 0)[0]
        #return  columnops.column_empty(0, in_index.dtype, False)
    cdef gdf_column** c_in_cols = cols_view_from_cols(in_cols)
    cdef cudf_table* c_in_table = new cudf_table(c_in_cols, col_count)

    cdef bool keep_first = (keep=='first')

    # check subset == in_cols and subset=None cases
    cdef gdf_column** key_cols
    cdef cudf_table* key_table
    if subset == in_cols or subset is None:
        key_cols = c_in_cols
        key_table = c_in_table
    else:
        key_cols = cols_view_from_cols(subset)
        key_table = new cudf_table(key_cols, len(subset))

    uniq_inds = rmm.device_array(len(in_cols[0]), dtype=np.int32)
    cdef uintptr_t c_uniq_ptr = get_ctype_ptr(uniq_inds)
    c_uniq = <gdf_index_type*>c_uniq_ptr
    unique_count=0
    with nogil:
        unique_count = gdf_get_unique_ordered_indices(key_table[0], 
                                                      c_uniq,
                                                      keep_first) 

    out_inds = clone_columns_with_size(in_index, unique_count)
    cdef gdf_column** c_out_inds = cols_view_from_cols(out_inds)
    cdef cudf_table* c_out_inds_table = new cudf_table(c_out_inds, 1)
    cdef gdf_column** c_in_inds = cols_view_from_cols(in_index)
    cdef cudf_table* c_in_inds_table = new cudf_table(c_in_inds, 1)
    
    if unique_count != 0 :
        with nogil:
            gather(c_in_inds_table, c_uniq, c_out_inds_table)

    if subset == in_cols or subset is None:
        pass
    else:
        free_table(key_table, key_cols)
    free_table(c_in_table, c_in_cols)

    free_table(c_in_inds_table, c_in_inds)
    free_table(c_out_inds_table, c_out_inds)

    return out_inds[0]


