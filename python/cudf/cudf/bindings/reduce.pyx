# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *

import numpy as np
import pandas as pd
import pyarrow as pa

from librmm_cffi import librmm as rmm

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.map cimport map as cmap
from libcpp.string  cimport string as cstring


pandas_version = tuple(map(int, pd.__version__.split('.', 2)[:2]))

_REDUCTION_OP = {
    'max': MAX,
    'min': MIN,
    'sum': SUM,
    'product': PRODUCT,
    'sum_of_squares': SUMOFSQUARES,
    'mean': MEAN,
    'var': VAR,
    'std': STD,
}

_SCAN_OP = {
    'sum': GDF_SCAN_SUM,
    'min': GDF_SCAN_MIN,
    'max': GDF_SCAN_MAX,
    'product': GDF_SCAN_PRODUCT,
}


def apply_reduce(reduction_op, col, dtype=None, ddof=1):
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

    check_gdf_compatibility(col)

    # check empty case
    if col.data.size <= col.null_count:
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
    cdef operators c_op = _REDUCTION_OP[reduction_op]
    cdef gdf_dtype c_out_dtype = gdf_dtype_from_value(col, col_dtype)
    cdef gdf_scalar c_result
    cdef gdf_size_type c_ddof = ddof

    with nogil:
        c_result = reduce(
            <gdf_column*>c_col,
            c_op,
            c_out_dtype,
            c_ddof
        )

    free_column(c_col)
    result = get_scalar_value(c_result, col_dtype)

    return result


def apply_scan(col_inp, col_out, scan_op, inclusive):
    """
      Call gdf scan.
    """

    check_gdf_compatibility(col_inp)
    check_gdf_compatibility(col_out)

    cdef gdf_column* c_col_inp = column_view_from_column(col_inp)
    cdef gdf_column* c_col_out = column_view_from_column(col_out)
    cdef gdf_scan_op c_op = _SCAN_OP[scan_op]
    cdef bool b_inclusive = <bool>inclusive

    with nogil:
        scan(
            <gdf_column*>c_col_inp,
            <gdf_column*>c_col_out,
            c_op,
            b_inclusive
        )

    free_column(c_col_inp)
    free_column(c_col_out)

    return
