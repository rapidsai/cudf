# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.dataframe import columnops
from cudf.dataframe.buffer import Buffer
from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.copying cimport *
import cudf.utils.utils as utils
from cudf.bindings.utils cimport columns_from_table, table_from_columns
from librmm_cffi import librmm as rmm

import numba
import numpy as np
import pandas as pd
import pyarrow as pa

from librmm_cffi import librmm as rmm

from libc.stdint cimport uintptr_t

from libcpp.map cimport map as cmap
from libcpp.string  cimport string as cstring


pandas_version = tuple(map(int, pd.__version__.split('.', 2)[:2]))


def clone_columns_with_size(in_cols, row_size):
    out_cols = []
    for col in in_cols:
        o_col = columnops.column_empty_like(col,
                                            dtype=col.dtype,
                                            masked=col.has_null_mask,
                                            newsize=row_size)
        out_cols.append(o_col)

    return out_cols


def _normalize_maps(maps, size):
    maps = columnops.as_column(maps).astype("int32")
    maps = maps.binary_operator("mod", np.int32(size))
    maps = maps.data.mem
    return maps


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

    maps = _normalize_maps(maps, in_size)

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
        c_out_cols = c_in_cols
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


def apply_scatter(source, maps, target):
    cdef cudf_table* c_source_table
    cdef cudf_table* c_target_table
    cdef cudf_table c_result_table
    cdef uintptr_t c_maps_ptr
    cdef gdf_index_type* c_maps

    source_cols = source
    target_cols = target

    if not isinstance(target_cols, (list, tuple)):
        target_cols = [target_cols]

    if not isinstance(source_cols, (list, tuple)):
        source_cols = [source_cols] * len(target_cols)

    for i in range(len(target_cols)):
        target_cols[i] = columnops.as_column(target_cols[i])
        source_cols[i] = columnops.as_column(source_cols[i])
        assert source_cols[i].dtype == target_cols[i].dtype

    c_source_table = table_from_columns(source_cols)
    c_target_table = table_from_columns(target_cols)

    maps = _normalize_maps(maps, len(target_cols[0]))

    c_maps_ptr = get_ctype_ptr(maps)
    c_maps = <gdf_index_type*>c_maps_ptr

    with nogil:
        c_result_table = scatter(
            c_source_table[0],
            c_maps,
            c_target_table[0])

    result_cols = columns_from_table(&c_result_table)

    del c_source_table
    del c_target_table

    if isinstance(target, (list, tuple)):
        return result_cols
    else:
        return result_cols[0]


def copy_column(input_col):
    """
        Call cudf::copy
    """
    cdef gdf_column* c_input_col = column_view_from_column(input_col)
    cdef gdf_column c_out_col

    with nogil:
        c_out_col = copy(c_input_col[0])

    free_column(c_input_col)

    return gdf_column_to_column(&c_out_col)


def apply_copy_range(out_col, in_col, int out_begin, int out_end,
                     int in_begin):
    from cudf.dataframe.column import Column

    if abs(out_end - out_begin) <= 1:
        return out_col

    if out_begin < 0:
        out_begin = len(out_col) + out_begin
    if out_end < 0:
        out_end = len(out_col) + out_end

    if out_begin > out_end:
        return out_col

    if out_col.null_count == 0 and in_col.has_null_mask:
        mask = utils.make_mask(len(out_col))
        cudautils.fill_value(mask, 0xff)
        out_col._mask = Buffer(mask)
        out_col._null_count = 0

    if in_col.null_count == 0 and out_col.has_null_mask:
        mask = utils.make_mask(len(in_col))
        cudautils.fill_value(mask, 0xff)
        in_col._mask = Buffer(mask)
        in_col._null_count = 0

    cdef gdf_column* c_out_col = column_view_from_column(out_col)
    cdef gdf_column* c_in_col = column_view_from_column(in_col)

    with nogil:
        copy_range(c_out_col,
                   c_in_col[0],
                   out_begin,
                   out_end,
                   in_begin)

    out_col._update_null_count(c_out_col.null_count)

    if out_col.dtype == np.dtype("object") and len(out_col) > 0:
        update_nvstrings_col(
            out_col,
            <uintptr_t>c_out_col.dtype_info.category)

    free_column(c_in_col)
    free_column(c_out_col)

    return out_col
