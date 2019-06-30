# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.utils.cudautils import astype, modulo
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


def clone_columns_with_size(in_cols, row_size):
    from cudf.dataframe import columnops
    out_cols = []
    for col in in_cols:
        o_col = columnops.column_empty(row_size,
                                       dtype = col.dtype,
                                       masked = col.has_null_mask)
        out_cols.append(o_col)

    return out_cols


def apply_gather(in_cols, maps, out_cols=None):
    """
      Call cudf::gather.

     * in_cols input column array
     * maps RMM device array with gdf_index_type (np.int32 compatible dtype)
     * out_cols the destination column array to output

     * returns out_cols
    """
    if in_cols[0].dtype == np.dtype("object"):
        in_size = in_cols[0].data.size()
    else:
        in_size = in_cols[0].data.size

    maps = astype(maps, 'int32')
    # TODO: replace with libcudf pymod when available
    maps = modulo(maps, in_size)

    col_count=len(in_cols)
    gather_count = len(maps)

    cdef gdf_column** c_in_cols = cols_view_from_cols(in_cols)
    cdef cudf_table* c_in_table = new cudf_table(c_in_cols, col_count)

    # check out_cols == in_cols and out_cols=None cases
    cdef bool is_same_input = False
    cdef gdf_column** c_out_cols
    cdef cudf_table* c_out_table
    if out_cols == in_cols :
        is_same_input = True
        c_out_table = c_in_table
    elif out_cols != None :
        c_out_cols = cols_view_from_cols(out_cols)
        c_out_table = new cudf_table(c_out_cols, col_count)
    else:
        out_cols = clone_columns_with_size(in_cols, gather_count)
        c_out_cols = cols_view_from_cols(out_cols)
        c_out_table = new cudf_table(c_out_cols, col_count)

    cdef uintptr_t c_maps_ptr
    cdef gdf_index_type* c_maps
    if gather_count != 0 :
        if out_cols[0].dtype == np.dtype("object"):
            out_size = out_cols[0].data.size()
        else:
            out_size = out_cols[0].data.size
        assert gather_count == out_size

        c_maps_ptr = get_ctype_ptr(maps)
        c_maps = <gdf_index_type*>c_maps_ptr

        with nogil:
            gather(c_in_table, c_maps, c_out_table)

    for i, col in enumerate(out_cols):
        col._update_null_count(c_out_cols[i].null_count)
        if col.dtype == np.dtype("object") and len(col) > 0:
            update_nvstrings_col(
                out_cols[i],
                <uintptr_t>c_out_cols[i].dtype_info.category)

    if is_same_input == False:
        free_table(c_out_table, c_out_cols)

    free_table(c_in_table, c_in_cols)

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


def apply_gather_array(dev_array, maps, out_col=None):
    """
      Call cudf::gather.

     * dev_array input device array
     * maps device array
     * out_cols the destination column to output

     * returns out_col
    """
    from cudf.dataframe import columnops

    in_col = columnops.as_column(dev_array)
    return apply_gather_column(in_col, maps, out_col)


def apply_scatter(in_cols, maps, out_cols=None):
    """
      Call cudf::scatter.

     * in_cols input column array
     * maps RMM device array with gdf_index_type (np.int32 compatible dtype)
     * out_cols the destination column array to output

     * returns out_cols
    """
    # TODO check the dtype of `maps` is compatible with gdf_index_type

    col_count=len(in_cols)
    gather_count = len(maps)
    cdef gdf_column** c_in_cols = cols_view_from_cols(in_cols)
    cdef cudf_table* c_in_table = new cudf_table(c_in_cols, col_count)

    # check out_cols == in_cols and out_cols=None cases
    cdef bool is_same_input = False
    cdef gdf_column** c_out_cols
    cdef cudf_table* c_out_table
    if out_cols == in_cols :
        is_same_input = True
        c_out_table = c_in_table
    elif out_cols != None :
        c_out_cols = cols_view_from_cols(out_cols)
        c_out_table = new cudf_table(c_out_cols, col_count)
    else:
        out_cols = clone_columns_with_size(in_cols, gather_count)
        c_out_cols = cols_view_from_cols(out_cols)
        c_out_table = new cudf_table(c_out_cols, col_count)

    cdef uintptr_t c_maps_ptr
    cdef gdf_index_type* c_maps
    if gather_count != 0 :
        # size check, cudf::gather requires same length for maps and in table.
        if out_cols[0].dtype == np.dtype("object"):
            in_size = in_cols[0].data.size()
        else:
            in_size = in_cols[0].data.size
        assert gather_count == in_size

        c_maps_ptr = get_ctype_ptr(maps)
        c_maps = <gdf_index_type*>c_maps_ptr

        with nogil:
            scatter(c_in_table, c_maps, c_out_table)

    if is_same_input == False :
        free_table(c_out_table, c_out_cols)

    free_table(c_in_table, c_in_cols)

    return out_cols


def apply_scatter_column(in_col, maps, out_col=None):
    """
      Call cudf::scatter.

     * in_cols input column
     * maps device array
     * out_cols the destination column to output

     * returns out_col
    """

    in_cols = [in_col]
    out_cols = None if out_col == None else [out_col]

    out_cols = apply_scatter(in_cols, maps, out_cols)

    return out_cols[0]


def apply_scatter_array(dev_array, maps, out_col=None):
    """
      Call cudf::scatter.

     * dev_array input device array
     * maps device array
     * out_cols the destination column to output

     * returns out_col
    """
    from cudf.dataframe import columnops

    in_col = columnops.as_column(dev_array)
    return apply_scatter_column(in_col, maps, out_col)
