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


cpdef replace(col, old_values, new_values):
    """
        Call gdf_find_and_replace_all
    """
    cdef gdf_column* c_col = column_view_from_column(col)
    cdef gdf_column* c_old_values = column_view_from_column(old_values)
    cdef gdf_column* c_new_values = column_view_from_column(new_values)

    gdf_find_and_replace_all(c_col, c_old_values, c_new_values)

cpdef replace_nulls(col, fill_values):
    """
        Call gdf_replace_nulls
    """
    cdef gdf_column* c_col = column_view_from_column(col)
    cdef gdf_column* fill_values_col = column_view_from_column(fill_values)

    gdf_replace_nulls(c_col, fill_values_col)
