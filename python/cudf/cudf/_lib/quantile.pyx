# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf.bindings.quantile import *
from cudf.bindings.utils cimport *

import numpy as np
import pandas as pd
import pyarrow as pa

from librmm_cffi import librmm as rmm
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libcpp.map cimport map as cmap
from libcpp.string  cimport string as cstring
from libcpp.utility cimport pair


pandas_version = tuple(map(int, pd.__version__.split('.', 2)[:2]))

_GDF_QUANTILE_METHODS = {
    'linear': GDF_QUANT_LINEAR,
    'lower': GDF_QUANT_LOWER,
    'higher': GDF_QUANT_HIGHER,
    'midpoint': GDF_QUANT_MIDPOINT,
    'nearest': GDF_QUANT_NEAREST,
}


def get_quantile_method(method):
    """Util to convert method to gdf gdf_quantile_method.
    """
    return _GDF_QUANTILE_METHODS[method]


def quantile(column, quant, method, exact):
    """ Calculate the `quant` quantile for the column
    Returns value with the quantile specified by quant
    """

    cdef gdf_column* c_col = column_view_from_column(column)
    cdef gdf_context* ctx = create_context_view(
        0,
        'sort',
        0,
        0,
        0,
        'null_as_largest',
        False
    )

    res = []
    cdef gdf_scalar* c_result = <gdf_scalar*>malloc(sizeof(gdf_scalar))
    for q in quant:
        if exact:
            gdf_quantile_exact(
                <gdf_column*>c_col,
                get_quantile_method(method),
                q,
                c_result,
                ctx
            )
        else:
            gdf_quantile_approx(
                <gdf_column*>c_col,
                q,
                c_result,
                ctx
            )
        if c_result.is_valid is True:
            res.append(get_scalar_value(c_result[0], column.dtype))

    free(c_result)
    free(ctx)

    return res

def apply_group_quantile(key_columns, value_columns, quant, method):
    """ Calculate the group wise `quant` quantile for the value_columns
    Returns column of group wise quantile specified by quant
    """

    cdef cudf_table *c_t = table_from_columns(key_columns)
    cdef cudf_table *c_val = table_from_columns(value_columns)
    if np.isscalar(quant):
        quant = [quant]
    cdef vector[double] q = quant
    cdef gdf_quantile_method c_interpolation = get_quantile_method(method)

    cdef pair[cudf_table, cudf_table] c_result
    with nogil:
        c_result = group_quantiles(c_t[0],
                                   c_val[0],
                                   q,
                                   c_interpolation,
                                   False)

    result_key_cols = columns_from_table(&c_result.first)
    result_val_cols = columns_from_table(&c_result.second)

    free(c_t)
    free(c_val)

    return (result_key_cols, result_val_cols)
