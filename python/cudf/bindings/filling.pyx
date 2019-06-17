# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np
from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.types import *
from cudf.bindings.filling import *
from cython.operator import dereference as deref
from libc.stdlib cimport free


def apply_scalar_fill(column, value, idx_begin, idx_end):
    """ Call cudf::fill

    Parameters
    ---------
    column : Column
        Column to fill
    value : scalar
        Scalar value to fill column with
    idx_begin, idx_end : gdf_index_type
        Beginning and end indices to fill between
    """
    check_gdf_compatibility(column)
    cdef gdf_column *out_col = column_view_from_column(column)

    cdef gdf_scalar *fill_value = gdf_scalar_from_scalar(np.array(value))
    cdef gdf_index_type c_idx_begin = <gdf_index_type>idx_begin
    cdef gdf_index_type c_idx_end = <gdf_index_type>idx_end

    fill(out_col, deref(fill_value), c_idx_begin, c_idx_end)

    free(out_col)
    free(fill_value)
