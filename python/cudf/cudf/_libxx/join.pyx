# cudf/_libxx/copying.pyx

import pandas as pd

from cudf._libxx.lib cimport *
from cudf._libxx.table cimport Table
from cudf._libxx.join cimport *

from cudf._lib.utils cimport *
from cudf._lib.utils import *


cpdef join(lhs, rhs, left_on, right_on, how, method):
    """
      Call libcudf++ join for full outer, inner and left joins.
      Returns a list of tuples [(column, valid, name), ...]
    """
    if how not in ['left', 'inner', 'outer', 'leftanti']:
        msg = "new join api only supports left, inner or outer"
        raise ValueError(msg)

    left_idx = []
    right_idx = []

    assert(len(left_on) == len(right_on))

    cdef vector[int] left_on_ind
    cdef vector[int] right_on_ind
    cdef vector[pair[int, int]] columns_in_common

    result_col_names = []  # Preserve the order of the column names

    for name, col in lhs._data.items():
        check_gdf_compatibility(col)
        result_col_names.append(name)

    for name in left_on:
        # This will ensure that the column name is valid
        lhs._data[name]
        left_on_ind.push_back(list(lhs._data.keys()).index(name))
        if (name in right_on and
           (left_on.index(name) == right_on.index(name))):
            columns_in_common.push_back(pair[int, int](
                list(lhs._data.keys()).index(name),
                list(rhs._data.keys()).index(name)))

    for name in right_on:
        # This will ensure that the column name is valid
        rhs._data[name]
        right_on_ind.push_back(list(lhs._data.keys()).index(name))

    for name, col in rhs._data.items():
        check_gdf_compatibility(col)
        if not ((name in left_on) and (name in right_on)
           and (left_on.index(name) == right_on.index(name))):
            result_col_names.append(name)

    cdef Table result

    if how == 'inner':
        result = inner_join(
            lhs, 
            rhs, 
            left_on_ind,
            right_on_ind,
            columns_in_common,
            result_col_names,
            lhs._index._column_names
        )
    elif how == 'left':
        result = left_join(
            lhs, 
            rhs, 
            left_on_ind,
            right_on_ind,
            columns_in_common,
            result_col_names,
            lhs._index._column_names
        )
    elif how == 'full':
        result = full_join(
            lhs, 
            rhs, 
            left_on_ind,
            right_on_ind,
            columns_in_common,
            result_col_names,
            lhs._index._column_names
        )

    return result

def inner_join(Table left, Table right, vector[int] left_on, vector[int] right_on, vector[pair[int, int]] columns_in_common, result_cols, result_index_cols):
    cdef unique_ptr[table] c_result = (
        cpp_inner_join(
            left.view(),
            right.view(),
            left_on,
            right_on,
            columns_in_common
        )
    )
    return Table.from_unique_ptr(
        move(c_result),
        column_names = result_cols,
        index_names = result_index_cols
    )

def left_join(Table left, Table right, vector[int] left_on, vector[int] right_on, vector[pair[int, int]] columns_in_common, result_cols, result_index_cols):
    cdef unique_ptr[table] c_result = (
        cpp_left_join(
            left.view(),
            right.view(),
            left_on,
            right_on,
            columns_in_common
        )
    )
    return Table.from_unique_ptr(
        move(c_result),
        column_names = result_cols,
        index_names = result_index_cols
    )

def full_join(Table left, Table right, vector[int] left_on, vector[int] right_on, vector[pair[int, int]] columns_in_common, result_cols, result_index_cols):
    cdef unique_ptr[table] c_result = (
        cpp_full_join(
            left.view(),
            right.view(),
            left_on,
            right_on,
            columns_in_common
        )
    )
    return Table.from_unique_ptr(
        move(c_result),
        column_names = result_cols,
        index_names = result_index_cols
    )