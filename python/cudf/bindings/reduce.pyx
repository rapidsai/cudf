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

cimport numpy as np

from librmm_cffi import librmm as rmm


from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.map cimport map as cmap
from libcpp.string  cimport string as cstring



# Cython function references need to be stored in a std::map
ctypedef gdf_error (*reduce_type)(gdf_column*, void*, gdf_size_type)

cdef cmap[cstring, reduce_type] _REDUCE_FUNCTIONS
_REDUCE_FUNCTIONS[b'max'] = gdf_max
_REDUCE_FUNCTIONS[b'min'] = gdf_min
_REDUCE_FUNCTIONS[b'sum'] = gdf_sum
_REDUCE_FUNCTIONS[b'sum_of_squares'] = gdf_sum_of_squares


def apply_reduce(reduction, col):
    """
      Call gdf reductions.
    """

    func = bytes(reduction, encoding="UTF-8")

    outsz = gdf_reduce_optimal_output_size()
    out = rmm.device_array(outsz, dtype=col.dtype)
    cdef uintptr_t out_ptr = get_ctype_ptr(out)

    check_gdf_compatibility(col)
    cdef gdf_column* c_col = column_view_from_column(col)

    cdef gdf_error result = _REDUCE_FUNCTIONS[func](<gdf_column*>c_col,
                            <void*>out_ptr,
                            outsz)

    check_gdf_error(result)

    free(c_col)


    return out[0]
