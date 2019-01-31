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


cpdef replace(col, old_values, new_values):
    """
        Call gdf_find_and_replace_all
    """
    cdef gdf_column* c_col = column_view_from_column(col)
    cdef gdf_column* c_old_values = column_view_from_column(old_values)
    cdef gdf_column* c_new_values = column_view_from_column(new_values)

    gdf_find_and_replace_all(c_col, c_old_values, c_new_values)
