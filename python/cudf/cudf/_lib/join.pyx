# Copyright (c) 2020, NVIDIA CORPORATION.

from collections import OrderedDict
from itertools import chain

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table, columns_from_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
cimport cudf._lib.cpp.join as cpp_join


cpdef join(Table lhs,
           Table rhs,
           object how,
           object method,
           object left_on=None,
           object right_on=None,
           bool left_index=False,
           bool right_index=False
           ):
    """
    Call libcudf++ join for full outer, inner and left joins.
    """
    cdef Table c_lhs = lhs
    cdef Table c_rhs = rhs

    # Views might or might not include index
    cdef table_view lhs_view
    cdef table_view rhs_view

    # Will hold the join column indices into L and R tables
    cdef vector[int] left_on_ind
    cdef vector[int] right_on_ind

    # If left/right index, will pass a full view
    # must offset the data column indices by # of index columns
    num_inds_left = len(left_on) + (lhs._num_indices * left_index)
    num_inds_right = len(right_on) + (rhs._num_indices * right_index)
    left_on_ind.reserve(num_inds_left)
    right_on_ind.reserve(num_inds_right)

    # Only used for semi or anti joins
    # The result columns are only the left hand columns
    cdef vector[int] all_left_inds = range(
        lhs._num_columns + (lhs._num_indices * left_index)
    )

    if left_index or right_index:
        # If either true, we need to process both indices as columns
        lhs_view = c_lhs.view()
        rhs_view = c_rhs.view()

        left_join_cols = list(lhs._index_names) + list(lhs._data.keys())
        right_join_cols = list(rhs._index_names) + list(rhs._data.keys())

        if left_index and right_index:
            # Index columns will be common, on the left, dropped from right
            # Index name is from the left
            # Both views, must take index column indices
            left_on_indices = right_on_indices = range(lhs._num_indices)
        elif left_index:
            # Joins left index columns with right 'on' columns
            left_on_indices = range(lhs._num_indices)
            right_on_indices = [
                right_join_cols.index(on_col) for on_col in right_on
            ]
        elif right_index:
            # Joins right index columns with left 'on' columns
            right_on_indices = range(rhs._num_indices)
            left_on_indices = [
                left_join_cols.index(on_col) for on_col in left_on
            ]
        for i_l, i_r in zip(left_on_indices, right_on_indices):
            left_on_ind.push_back(i_l)
            right_on_ind.push_back(i_r)
    else:
        # cuDF's Python layer will create a new RangeIndex for this case
        lhs_view = c_lhs.data_view()
        rhs_view = c_rhs.data_view()

        left_join_cols = list(lhs._data.keys())
        right_join_cols = list(rhs._data.keys())

    # If both left/right_index, joining on indices plus additional cols
    # If neither, joining on just cols, not indices
    # In both cases, must match up additional column indices in lhs/rhs
    if left_index == right_index:
        for name in left_on:
            left_on_ind.push_back(left_join_cols.index(name))
        for name in right_on:
            right_on_ind.push_back(right_join_cols.index(name))

    cdef pair[unique_ptr[column], unique_ptr[column]] c_result
    if how == 'inner':
        with nogil:
            c_result = move(cpp_join.inner_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
            ))
    elif how == 'left':
        with nogil:
            c_result = move(cpp_join.left_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
            ))
    elif how == 'outer':
        with nogil:
            c_result = move(cpp_join.full_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
            ))
    return (
        Column.from_unique_ptr(move(c_result.first)),
        Column.from_unique_ptr(move(c_result.second))
    )
