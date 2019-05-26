# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.replace cimport *
from cudf.dataframe.column import Column

import numpy as np
import pandas as pd
import pyarrow as pa

from librmm_cffi import librmm as rmm

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free


cpdef replace(input_col, values_to_replace, replacement_values):
    """
        Call cudf::find_and_replace_all
    """
    cdef gdf_column* c_input_col = \
                            column_view_from_column(input_col)
    cdef gdf_column* c_values_to_replace = \
                            column_view_from_column(values_to_replace)
    cdef gdf_column* c_replacement_values = \
                            column_view_from_column(replacement_values)

    cdef gdf_column output

    with nogil:
        output = find_and_replace_all(c_input_col[0],
                                      c_values_to_replace[0],
                                      c_replacement_values[0])

    data, mask = gdf_column_to_column_mem(&output)

    return Column.from_mem_views(data, mask)


cpdef replace_nulls(col, fill_values):
    """
        Call gdf_replace_nulls
    """
    cdef gdf_column* c_col = column_view_from_column(col)
    cdef gdf_column* fill_values_col = column_view_from_column(fill_values)

    gdf_replace_nulls(c_col, fill_values_col)
