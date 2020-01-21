# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libcpp.map cimport map as cmap
from libcpp.string  cimport string as cstring
from libcpp.utility cimport pair

import rmm

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._libxx.column cimport Column
from cudf._lib.utils cimport *
cimport cudf._lib.includes.reduce as cpp_reduce

pandas_version = tuple(map(int, pd.__version__.split('.', 2)[:2]))

_REDUCTION_OP = {
    'max': cpp_reduce.MAX,
    'min': cpp_reduce.MIN,
    'any': cpp_reduce.ANY,
    'all': cpp_reduce.ALL,
    'sum': cpp_reduce.SUM,
    'product': cpp_reduce.PRODUCT,
    'sum_of_squares': cpp_reduce.SUMOFSQUARES,
    'mean': cpp_reduce.MEAN,
    'var': cpp_reduce.VAR,
    'std': cpp_reduce.STD,
}

_SCAN_OP = {
    'sum': cpp_reduce.GDF_SCAN_SUM,
    'min': cpp_reduce.GDF_SCAN_MIN,
    'max': cpp_reduce.GDF_SCAN_MAX,
    'product': cpp_reduce.GDF_SCAN_PRODUCT,
}


def reduce(reduction_op, Column col, dtype=None, ddof=1):
    """
      Call gdf reductions.

    Args:
        reduction_op: reduction operator as string. It should be one of
        'min', 'max', 'sum', 'product', 'sum_of_squares', 'mean', 'var', 'std'
        col: input column to apply reduction operation on
        dtype: output dtype
        ddof: This parameter is used only for 'std' and 'var'.
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.

    Returns:
        dtype scalar value of reduction operation on column
    """
    # check empty case
    if len(col) <= col.null_count:
        if reduction_op == 'sum' or reduction_op == 'sum_of_squares':
            return col.dtype.type(0)
        if reduction_op == 'product' and pandas_version >= (0, 22):
            return col.dtype.type(1)
        return np.nan

    col_dtype = col.dtype
    if reduction_op in ['sum', 'sum_of_squares', 'product']:
        col_dtype = np.find_common_type([col_dtype], [np.int64])
    col_dtype = col_dtype if dtype is None else dtype

    cdef gdf_column* c_col = column_view_from_column(col)
    cdef gdf_dtype c_out_dtype = gdf_dtype_from_dtype(col_dtype)
    cdef gdf_scalar c_result
    cdef size_type c_ddof = ddof
    cdef cpp_reduce.operators c_op = _REDUCTION_OP[reduction_op]

    with nogil:
        c_result = cpp_reduce.reduce(
            <gdf_column*>c_col,
            c_op,
            c_out_dtype,
            c_ddof
        )

    free_column(c_col)
    result = get_scalar_value(c_result, col_dtype)

    return result


def scan(col_inp, col_out, scan_op, inclusive):
    """
      Call gdf scan.
    """

    check_gdf_compatibility(col_inp)
    check_gdf_compatibility(col_out)

    cdef gdf_column* c_col_inp = column_view_from_column(col_inp)
    cdef gdf_column* c_col_out = column_view_from_column(col_out)
    cdef cpp_reduce.gdf_scan_op c_op = _SCAN_OP[scan_op]
    cdef bool b_inclusive = <bool>inclusive

    with nogil:
        cpp_reduce.scan(
            <gdf_column*>c_col_inp,
            <gdf_column*>c_col_out,
            c_op,
            b_inclusive
        )

    free_column(c_col_inp)
    free_column(c_col_out)

    return


def group_std(key_columns, value_columns, ddof=1):
    """ Calculate the group wise `quant` quantile for the value_columns
    Returns column of group wise quantile specified by quant
    """

    cdef cudf_table *c_t = table_from_columns(key_columns)
    cdef cudf_table *c_val = table_from_columns(value_columns)
    cdef size_type c_ddof = ddof

    cdef pair[cudf_table, cudf_table] c_result
    with nogil:
        c_result = cpp_reduce.group_std(
            c_t[0],
            c_val[0],
            c_ddof
        )

    result_key_cols = columns_from_table(&c_result.first)
    result_val_cols = columns_from_table(&c_result.second)

    free(c_t)
    free(c_val)

    return (result_key_cols, result_val_cols)
