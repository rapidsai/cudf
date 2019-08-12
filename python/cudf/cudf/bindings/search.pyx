# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.search cimport *
from cudf.bindings.utils cimport *
from cudf.bindings.utils import *
from libcpp.vector cimport vector
from libc.stdlib cimport free


def search_sorted(column, values, side):
    """Find indices where elements should be inserted to maintain order

    Parameters
    ----------
    column : Column
        Column to search in
    values : Column
        Column of values to search for
    side : str {‘left’, ‘right’} optional
        If ‘left’, the index of the first suitable location found is given.
        If ‘right’, return the last such index
    """
    cdef cudf_table *c_t = table_from_columns([column])
    cdef cudf_table *c_values = table_from_columns([values])

    cdef vector[bool] c_desc_flags
    c_desc_flags.push_back(False)

    cdef gdf_column result

    if side == 'left':
        with nogil:
            result = lower_bound(c_t[0], c_values[0], c_desc_flags)
    if side == 'right':
        with nogil:
            result = upper_bound(c_t[0], c_values[0], c_desc_flags)

    free_table(c_t)
    free_table(c_values)

    return gdf_column_to_column(&result)
