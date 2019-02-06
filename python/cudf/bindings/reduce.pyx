# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from .cudf_cpp cimport *
from .cudf_cpp import *

import numpy as np
import pandas as pd
import pyarrow as pa
pandas_version = tuple(map(int,pd.__version__.split('.', 2)[:2]))


cimport numpy as np

from librmm_cffi import librmm as rmm


from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.map cimport map as cmap
from libcpp.string  cimport string as cstring



def apply_reduce(reduction, col):
    """
      Call gdf reductions.
    """


    outsz = gdf_reduce_optimal_output_size()
    out = rmm.device_array(outsz, dtype=col.dtype)
    cdef uintptr_t out_ptr = get_ctype_ptr(out)

    check_gdf_compatibility(col)
    cdef gdf_column* c_col = column_view_from_column(col)

    cdef gdf_error result
    with nogil:    
        if reduction == 'max':
            result = gdf_max(<gdf_column*>c_col, <void*>out_ptr, outsz)
        elif reduction == 'min':
            result = gdf_min(<gdf_column*>c_col, <void*>out_ptr, outsz)
        elif reduction == 'sum':
            result = gdf_sum(<gdf_column*>c_col, <void*>out_ptr, outsz)
        elif reduction == 'sum_of_squares':
            result = gdf_sum_of_squares(<gdf_column*>c_col,
                                        <void*>out_ptr,
                                        outsz)
        elif reduction == 'product':
            result = gdf_product(<gdf_column*>c_col, <void*>out_ptr, outsz)
        else:
            result = GDF_NOTIMPLEMENTED_ERROR

    free(c_col)

    if result == GDF_DATASET_EMPTY:
        if reduction == 'sum' or reduction == 'sum_of_squares':
            return col.dtype.type(0)
        if reduction == 'product' and pandas_version >= (0, 22):
            return col.dtype.type(1)
        return np.nan

    check_gdf_error(result)

    return out[0]
