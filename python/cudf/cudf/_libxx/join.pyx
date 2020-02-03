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
    if how not in ['left', 'inner', 'outer']:
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
        right_on_ind.push_back(list(rhs._data.keys()).index(name))

    for name, col in rhs._data.items():
        check_gdf_compatibility(col)
        if not ((name in left_on) and (name in right_on)
           and (left_on.index(name) == right_on.index(name))):
            result_col_names.append(name)

    cdef unique_ptr[table] c_result
    cdef Table c_lhs = lhs
    cdef Table c_rhs = rhs
    cdef table_view lhs_view = c_lhs.data_view()
    cdef table_view rhs_view = c_rhs.data_view()

    with nogil:
        if how == 'inner':
            c_result = move(cpp_inner_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                columns_in_common
            ))
        elif how == 'left':
            c_result = move(cpp_left_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                columns_in_common
            ))
        elif how == 'outer':
            c_result = move(cpp_full_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                columns_in_common
            ))
    py_result = Table.from_unique_ptr(
        move(c_result),
        column_names = result_col_names)
    return py_result
