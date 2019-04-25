# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.types cimport table as cudf_table
from cudf.bindings.types import *
from librmm_cffi import librmm as rmm

import numpy as np
import pandas as pd
import pyarrow as pa
pandas_version = tuple(map(int,pd.__version__.split('.', 2)[:2]))

from librmm_cffi import librmm as rmm

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.map cimport map as cmap
from libcpp.string  cimport string as cstring


cdef cudf_table* table_from_cols(cols):
    col_counts=len(cols)
    cdef gdf_column **c_cols = <gdf_column**>malloc(sizeof(gdf_column*)*col_count)

    cdef i
    for i in range(col_count):
        check_gdf_compatibility(cols[i])
        c_cols[i] = column_view_from_column(cols[i])

    cdef cudf_table* table  = new cudf_table(c_cols, col_count)
    return table


cdef free_table(cudf_table* table):
    cdef i
    cdef gdf_column *c_col
    for i in range(table[0].num_columns()) :
        c_col = table[0].get_column(i)
        free(c_col)

    del table


def clone_column_with_size(col, size):
    from cudf.dataframe import columnops
    return columnops.column_empty(size, col.dtype, cols.has_null_mask())


def clone_columns_with_size(in_cols, size):
    out_cols = []
    for col in in_cols:
        o_col = clone_column_with_size(col, size)
        out_cols.append(o_col)

    return out_cols


def apply_gather(in_cols, maps, out_cols=None):
    """
      Call cudf::gather.

     * in_cols input column array
     * maps RMM device array
     * out_cols the destination column array to output

     * returns out_cols
    """

    cdef cudf_table* c_in_table = table_from_cols(in_cols)

    cdef bool is_same_input = false
    cdef cudf_table* c_out_table
    if out_cols == in_cols :
        is_same_input = true
        c_out_table = c_in_table
    elif out_cols != None :
        c_out_table = table_from_cols(out_cols)
    else:
        out_cols = clone_table_with_size(in_cols, size)
        c_out_table = table_from_cols(out_cols)

    # size check, cudf::gather requires same length for maps and out table.
    assert len(maps) == len(out_cols)

    cdef uintptr_t c_maps_ptr
    cdef gdf_index_type* c_maps
    c_maps_ptr = get_ctype_ptr(maps)
    c_maps = <gdf_index_type*>c_maps_ptr

    #cdef void* void_ptr = <void*> get_ctype_ptr(maps)
    #cdef gdf_index_type* c_maps = <gdf_index_type*>void_ptr


    with nogil:
        gather(c_in_table, c_maps, c_out_table)

    if is_same_input == false :
        free_table(c_out_table)
    
    free_table(c_in_table)
    
    return out_cols


def apply_gather_column(in_col, maps, out_col=None):
    """
      Call cudf::gather.

     * in_cols input column
     * maps device array
     * out_cols the destination column to output

     * returns out_col
    """

    in_cols = [in_col]
    out_cols = None if out_col == None else [out_col]

    out_cols = apply_gather(in_cols, maps, out_cols)

    return out_cols[0]




