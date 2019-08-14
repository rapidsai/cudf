# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.issorted cimport *
from cudf.bindings.utils cimport *
from cudf.bindings.utils import *

cpdef issorted(columns, descending=[], nulls_are_smallest=False):
    """
    Checks whether the rows of a `table` are sorted in a lexicographical order.

    Parameters
    ----------
    columns : list of columns
    descending : array of order of columns, 0 - ascending, 1 - decending
                 if array passed is of zero length, by deafult all columns
                 are considered ascending. If this an empty vector, then it
                 will be assumed that each column is in ascending order.
    nulls_are_smallest : True indicates nulls are to be considered
                         smaller than non-nulls ; false indicates oppositie

    Returns
    -------
    result : True - if sorted; False - if not.
    """

    cdef cudf_table *c_values_table = table_from_columns(columns)
    cdef vector[int8_t] c_descending_vector = descending
    cdef bool c_nulls_are_smallest = null_are_smallest
    cdef bool c_result = False
    with nogil:
        c_result = is_sorted(c_values_table[0],
                             c_descending_vector, c_nulls_are_smallest)

    del c_values_table

    return c_result
