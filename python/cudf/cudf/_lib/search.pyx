# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._lib.utils cimport *
from cudf._lib.utils import *
from cudf._lib.includes cimport search as cpp_search
from libcpp.vector cimport vector


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

    cdef gdf_column c_out_col

    if side == 'left':
        with nogil:
            c_out_col = cpp_search.lower_bound(
                c_t[0],
                c_values[0],
                c_desc_flags
            )
    elif side == 'right':
        with nogil:
            c_out_col = cpp_search.upper_bound(
                c_t[0],
                c_values[0],
                c_desc_flags
            )

    free_table(c_t)
    free_table(c_values)

    return gdf_column_to_column(&c_out_col)


def contains (column, item):
    """Check whether column contains the value

    Parameters
    ----------
    column : NumericalColumn
        Column to search in
    item :
        value to be searched
    """
    if (len(column) == 0 or item is None):
        return False

    cdef gdf_column* col = column_view_from_column(column)
    cdef gdf_scalar* item_scalar = gdf_scalar_from_scalar(item)

    cdef bool result = cpp_search.contains(col[0], item_scalar[0])

    return result
