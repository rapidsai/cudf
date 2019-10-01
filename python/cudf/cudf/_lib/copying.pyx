# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.core.buffer import Buffer
from cudf._lib.cudf cimport *
from cudf._lib.cudf import *

import cudf.utils.utils as utils
from cudf.utils.dtypes import is_string_dtype
from cudf._lib.utils cimport (
    columns_from_table,
    table_from_columns,
    table_to_dataframe
)
import rmm
from cudf._lib.includes.copying cimport (
    copy as cpp_copy,
    copy_range as cpp_copy_range,
    gather as cpp_gather,
    scatter as cpp_scatter,
    scatter_to_tables as cpp_scatter_to_tables
)

import numba
import numpy as np
import pandas as pd
import pyarrow as pa

import rmm

from libc.stdint cimport uintptr_t

from libcpp.map cimport map as cmap
from libcpp.string  cimport string as cstring


pandas_version = tuple(map(int, pd.__version__.split('.', 2)[:2]))


def clone_columns_with_size(in_cols, row_size):
    from cudf.core.column import column

    out_cols = []
    for col in in_cols:
        o_col = column.column_empty_like(col,
                                         dtype=col.dtype,
                                         masked=col.has_null_mask,
                                         newsize=row_size)
        out_cols.append(o_col)

    return out_cols


def _normalize_maps(maps, size):
    from cudf.core.column import column

    maps = column.as_column(maps).astype("int32")
    maps = maps.binary_operator("mod", np.int32(size))
    maps = maps.data.mem
    return maps


def gather(source, maps, bounds_check=True):
    """
    Gathers elements from source into dest (if given) using the gathermap maps.
    If dest is not given, it is allocated inside the function and returned.

    Parameters
    ----------
    source : Column or list of Columns
    maps : DeviceNDArray

    Returns
    -------
    Column or list of Columns, or None if dest is given
    """
    from cudf.core.column import column, CategoricalColumn

    if isinstance(source, (list, tuple)):
        in_cols = source
    else:
        in_cols = [source]

    for i, in_col in enumerate(in_cols):
        in_cols[i] = column.as_column(in_cols[i])

    if is_string_dtype(in_cols[0]):
        in_size = in_cols[0].data.size()
    else:
        in_size = in_cols[0].data.size

    maps = column.as_column(maps)

    col_count=len(in_cols)
    gather_count = len(maps)

    cdef cudf_table* c_in_table = table_from_columns(in_cols)
    cdef cudf_table c_out_table
    cdef gdf_column* c_maps = column_view_from_column(maps)
    cdef bool c_bounds_check = bounds_check

    with nogil:
        c_out_table = cpp_gather(c_in_table, c_maps[0], c_bounds_check)

    out_cols = columns_from_table(&c_out_table)

    for i, in_col in enumerate(in_cols):
        if isinstance(in_col, CategoricalColumn):
            out_cols[i] = CategoricalColumn(
                data=out_cols[i].data,
                mask=out_cols[i].mask,
                categories=in_col.cat().categories,
                ordered=in_col.cat().ordered)

    free_column(c_maps)
    free_table(c_in_table)

    if isinstance(source, (list, tuple)):
        return out_cols
    else:
        return out_cols[0]


def scatter(source, maps, target, bounds_check=True):
    from cudf.core.column import column

    cdef cudf_table* c_source_table
    cdef cudf_table* c_target_table
    cdef cudf_table c_result_table

    source_cols = source
    target_cols = target

    if not isinstance(target_cols, (list, tuple)):
        target_cols = [target_cols]

    if not isinstance(source_cols, (list, tuple)):
        source_cols = [source_cols] * len(target_cols)

    for i in range(len(target_cols)):
        target_cols[i] = column.as_column(target_cols[i])
        source_cols[i] = column.as_column(source_cols[i])
        assert source_cols[i].dtype == target_cols[i].dtype

    c_source_table = table_from_columns(source_cols)
    c_target_table = table_from_columns(target_cols)

    cdef gdf_column* c_maps = column_view_from_column(maps)
    cdef bool c_bounds_check = bounds_check

    with nogil:
        c_result_table = cpp_scatter(
            c_source_table[0],
            c_maps[0],
            c_target_table[0],
            c_bounds_check)

    free_column(c_maps)

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
        c_out_col = cpp_copy(c_input_col[0])

    free_column(c_input_col)

    return gdf_column_to_column(&c_out_col)


def copy_range(out_col, in_col, int out_begin, int out_end,
               int in_begin):
    from cudf.core.column import Column

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
        cpp_copy_range(c_out_col,
                       c_in_col[0],
                       out_begin,
                       out_end,
                       in_begin)

    out_col._update_null_count(c_out_col.null_count)

    if is_string_dtype(out_col) and len(out_col) > 0:
        update_nvstrings_col(
            out_col,
            <uintptr_t>c_out_col.dtype_info.category)

    free_column(c_in_col)
    free_column(c_out_col)

    return out_col


def scatter_to_frames(source, maps):
    """
    Scatters rows to 'n' dataframes according to maps

    Parameters
    ----------
    source : Column or list of Columns
    maps : non-null column with values ranging from 0 to n-1 for each row

    Returns
    -------
    list of scattered dataframes
    """
    from cudf.core.column import column

    in_cols = source
    col_count=len(in_cols)
    if col_count == 0:
        return []
    for i, in_col in enumerate(in_cols):
        in_cols[i] = column.as_column(in_cols[i])

    if is_string_dtype(in_cols[0]):
        in_size = in_cols[0].data.size()
    else:
        in_size = in_cols[0].data.size

    maps = column.as_column(maps).astype("int32")
    gather_count = len(maps)
    assert(gather_count == in_size)

    cdef gdf_column** c_in_cols = cols_view_from_cols(in_cols)
    cdef cudf_table* c_in_table = new cudf_table(c_in_cols, col_count)
    cdef gdf_column* c_maps = column_view_from_column(maps)
    cdef vector[cudf_table] c_out_tables

    with nogil:
        c_out_tables = cpp_scatter_to_tables(c_in_table[0], c_maps[0])

    out_tables = []
    for tab in c_out_tables:
        out_tables.append(table_to_dataframe(&tab, int_col_names=False))

    free_table(c_in_table, c_in_cols)
    free_column(c_maps)

    return out_tables
