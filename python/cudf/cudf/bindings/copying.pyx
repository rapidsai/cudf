# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.dataframe import columnops
from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.utils.cudautils import astype, modulo
from librmm_cffi import librmm as rmm

import numba
import numpy as np
import pandas as pd
import pyarrow as pa

from librmm_cffi import librmm as rmm

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.map cimport map as cmap
from libcpp.string  cimport string as cstring


pandas_version = tuple(map(int, pd.__version__.split('.', 2)[:2]))


def clone_columns_with_size(in_cols, row_size):
    from cudf.dataframe import columnops
    out_cols = []
    for col in in_cols:
        o_col = columnops.column_empty_like(col,
                                            dtype=col.dtype,
                                            masked=col.has_null_mask,
                                            newsize=row_size)
        out_cols.append(o_col)

    return out_cols


def apply_gather(source, maps, dest=None):
    """
    Gathers elements from source into dest (if given) using the gathermap maps.
    If dest is not given, it is allocated inside the function and returned.

    Parameters
    ----------
    source : Column or list of Columns
    maps : DeviceNDArray
    dest : Column or list of Columns (optional)

    Returns
    -------
    Column or list of Columns, or None if dest is given
    """
    if isinstance(source, (list, tuple)):
        if dest is not None:
            assert(isinstance(dest, (list, tuple)))
        in_cols = source
        out_cols = dest
    else:
        in_cols = [source]
        out_cols = None if dest is None else [dest]

    for i, in_col in enumerate(in_cols):
        in_cols[i] = columnops.as_column(in_cols[i])
        if dest is not None:
            out_cols[i] = columnops.as_column(out_cols[i])
    
    if in_cols[0].dtype == np.dtype("object"):
        in_size = in_cols[0].data.size()
    else:
        in_size = in_cols[0].data.size

    maps = columnops.as_column(maps).astype("int32")
    maps = maps.binary_operator("mod", maps.normalize_binop_value(in_size))
    maps = maps.data.mem

    col_count=len(in_cols)
    gather_count = len(maps)

    cdef gdf_column** c_in_cols = cols_view_from_cols(in_cols)
    cdef cudf_table* c_in_table = new cudf_table(c_in_cols, col_count)

    # check out_cols == in_cols and out_cols=None cases
    cdef bool is_same_input = False
    cdef gdf_column** c_out_cols
    cdef cudf_table* c_out_table
    if out_cols == in_cols:
        is_same_input = True
        c_out_table = c_in_table
    elif out_cols is not None:
        c_out_cols = cols_view_from_cols(out_cols)
        c_out_table = new cudf_table(c_out_cols, col_count)
    else:
        out_cols = clone_columns_with_size(in_cols, gather_count)
        c_out_cols = cols_view_from_cols(out_cols)
        c_out_table = new cudf_table(c_out_cols, col_count)

    cdef uintptr_t c_maps_ptr
    cdef gdf_index_type* c_maps
    if gather_count != 0:
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

    if is_same_input is False:
        free_table(c_out_table, c_out_cols)

    free_table(c_in_table, c_in_cols)

    if dest is not None:
        return
    
    if isinstance(source, (list, tuple)):
        return out_cols
    else:
        return out_cols[0]


def copy_column(input_col):
    """
        Call cudf::copy
    """
    cdef gdf_column* c_input_col = column_view_from_column(input_col)
    cdef gdf_column* output = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        output[0] = copy(c_input_col[0])

    data, mask = gdf_column_to_column_mem(output)
    from cudf.dataframe.column import Column

    free(c_input_col)
    free(output)

    return Column.from_mem_views(data, mask, output.null_count)
