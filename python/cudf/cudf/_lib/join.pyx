# Copyright (c) 2020, NVIDIA CORPORATION.

from collections import OrderedDict
from itertools import chain

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

from cudf._lib.table cimport Table, columns_from_ptr

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
    cdef vector[int] all_right_inds = range(
        rhs._num_columns + (rhs._num_indices * right_index)
    )

    result_col_names = compute_result_col_names(lhs, rhs, how)

    columns_in_common = OrderedDict()
    cdef vector[pair[int, int]] c_columns_in_common

    # keep track of where the desired index column will end up
    result_index_pos = None
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
            result_idx_positions = range(lhs._num_indices)
            result_index_names = lhs._index_names

        elif left_index:
            # Joins left index columns with right 'on' columns
            left_on_indices = range(lhs._num_indices)
            right_on_indices = [
                right_join_cols.index(on_col) for on_col in right_on
            ]

            # The left index columns 'become' the new RHS columns
            # and the right index 'survives'
            result_idx_positions = range(
                len(left_join_cols), len(left_join_cols) + lhs._num_indices
            )
            result_index_names = rhs._index_names

            # but since the common columns are gathered from the left
            # the rhs 'on' cols are returned on the left of the result
            # rearrange the names so account for this
            common = [None] * rhs._num_indices
            for i in range(rhs._num_indices):
                common[i] = result_col_names.pop(
                    result_col_names.index(right_on[i])
                )
            result_col_names = common + result_col_names
        elif right_index:
            # Joins right index columns with left 'on' columns
            right_on_indices = range(rhs._num_indices)
            left_on_indices = [
                left_join_cols.index(on_col) for on_col in left_on
            ]

            # The right index columns 'become' the new LHS columns
            # and the left index survives
            # since they are already gathered from the left,
            # no rearranging has to be done
            result_idx_positions = range(lhs._num_indices)
            result_index_names = lhs._index_names
        for i_l, i_r in zip(left_on_indices, right_on_indices):
            left_on_ind.push_back(i_l)
            right_on_ind.push_back(i_r)
            columns_in_common[(i_l, i_r)] = None
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
            if name in right_on:
                if (left_on.index(name) == right_on.index(name)):
                    columns_in_common[(
                        left_join_cols.index(name),
                        right_join_cols.index(name)
                    )] = None
        for name in right_on:
            right_on_ind.push_back(right_join_cols.index(name))
    c_columns_in_common = list(columns_in_common.keys())
    cdef unique_ptr[table] c_result
    if how == 'inner':
        with nogil:
            c_result = move(cpp_join.inner_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                c_columns_in_common
            ))
    elif how == 'left':
        with nogil:
            c_result = move(cpp_join.left_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                c_columns_in_common
            ))
    elif how == 'outer':
        with nogil:
            c_result = move(cpp_join.full_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                c_columns_in_common
            ))
    elif how == 'leftsemi':
        with nogil:
            c_result = move(cpp_join.left_semi_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                all_left_inds
            ))
    elif how == 'leftanti':
        with nogil:
            c_result = move(cpp_join.left_anti_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                all_left_inds
            ))

    all_cols_py = columns_from_ptr(move(c_result))
    if left_index or right_index:
        ind_cols = OrderedDict()
        for name, pos in zip(
            result_index_names[::-1], result_idx_positions[::-1]
        ):
            ind_cols[name] = all_cols_py.pop(pos)
        index = OrderedDict()
        for k, v in reversed(ind_cols.items()):
            index[k] = v
        index = Table(index)
    else:
        index = None
    data_ordered_dict = OrderedDict(zip(result_col_names, all_cols_py))
    return Table(data=data_ordered_dict, index=index)


def compute_result_col_names(lhs, rhs, how):
    """
    Determine the names of the data columns in the result of
    a libcudf join, based on the original left and right frames
    as well as the type of join that was performed.
    """
    if how in {"left", "inner", "outer", "leftsemi", "leftanti"}:
        a = lhs._data.keys()
        if how not in {"leftsemi", "leftanti"}:
            return list(chain(a, (k for k in rhs._data.keys()
                        if k not in lhs._data.keys())))
        return list(a)
    else:
        raise NotImplementedError(
            f"{how} merge not supported yet"
        )
