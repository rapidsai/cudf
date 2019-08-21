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


def apply_quantile(column, quant, method, exact):
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
        'null_as_largest'
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
