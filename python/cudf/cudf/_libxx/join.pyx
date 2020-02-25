from cudf._libxx.lib cimport *
from cudf._libxx.table cimport Table, columns_from_ptr
from cudf._libxx.includes.join cimport *
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from collections import OrderedDict

cpdef join(Table lhs, Table rhs, object left_on, object right_on, object how, object method, bool left_index=False, bool right_index=False):
    """
    Call libcudf++ join for full outer, inner and left joins.
    Returns a list of tuples [(column, valid, name), ...]
    """

    cdef Table c_lhs = lhs
    cdef Table c_rhs = rhs

    num_inds_left = len(left_on) + left_index
    num_inds_right = len(right_on) + right_index
    cdef vector[int] left_on_ind
    cdef vector[int] right_on_ind
    left_on_ind.reserve(num_inds_left)
    right_on_ind.reserve(num_inds_right)

    cdef vector[pair[int, int]] columns_in_common

    # create the two views that will be set later to include or not include the index column 
    cdef table_view lhs_view
    cdef table_view rhs_view

    # the result columns are all the left columns (including common ones) 
    # + all the right columns (excluding the common ones)
    result_col_names = []

    for name in lhs._data.keys():
        result_col_names.append(name)
    for name in rhs._data.keys():
        if name not in lhs._data.keys():
            result_col_names.append(name)

    # keep track of where the desired index column will end up
    result_index_pos = None
    # if left_index or right_index is true, then both indices must be processed as join columns
    if left_index or right_index:
        # If one index is specified, it will be joined against some column from the other
        # the result will be returned as a gather from the other column, and the index itself drops
        # In addition, the opposite index ends up as the final index 
        # Thus if either are true, we need to process both indices as join columns
        lhs_view = c_lhs.view()
        rhs_view = c_rhs.view()

        left_join_cols = [lhs.index.name] + list(lhs._data.keys())
        right_join_cols = [rhs.index.name] + list(rhs._data.keys())
        if left_index and right_index:
            left_on_idx = right_on_idx = 0
            result_index_pos = 0
            result_index_name = rhs.index.name

        elif left_index:
            left_on_idx = 0
            right_on_idx = right_join_cols.index(right_on[0])
            result_index_pos = len(left_join_cols)
            result_index_name = rhs.index.name

            common = result_col_names.pop(result_col_names.index(right_on[0]))
            result_col_names = [common] + result_col_names
        elif right_index:
            right_on_idx = 0
            left_on_idx = left_join_cols.index(left_on[0])
            result_index_pos = 0
            result_index_name = lhs.index.name

        left_on_ind.push_back(left_on_idx)
        right_on_ind.push_back(right_on_idx)
        
        columns_in_common.push_back(pair[int,int]((left_on_idx, right_on_idx)))

    else:
        # If neither index is specified, we can exclude them from the join completely
        # cuDF's Python layer will create a new RangeIndex for this case
        lhs_view = c_lhs.data_view()
        rhs_view = c_rhs.data_view()

        left_join_cols = list(lhs._data.keys())
        right_join_cols = list(rhs._data.keys())

    # If only one index is specified, there will only be one join column from other
    # If neither are specified, we need to build libcudf arguments for the actual join columns
    # If both, could be joining on the indices as well as other common cols, so we must search
    if left_index == right_index:
        for name in left_on:
            left_on_ind.push_back(left_join_cols.index(name))
            if (name in right_on and
            (left_on.index(name) == right_on.index(name))):
                columns_in_common.push_back(
                    pair[int, int](
                    (
                        left_join_cols.index(name),
                        right_join_cols.index(name)
                )))
        for name in right_on:
            right_on_ind.push_back(right_join_cols.index(name))

    cdef unique_ptr[table] c_result
    if how == 'inner':
        with nogil:
            c_result = move(cpp_inner_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                columns_in_common
            ))
    elif how == 'left':
        with nogil:
            c_result = move(cpp_left_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                columns_in_common
            ))
    elif how == 'outer':
        with nogil:
            c_result = move(cpp_full_join(
                lhs_view,
                rhs_view,
                left_on_ind,
                right_on_ind,
                columns_in_common
            ))

    all_cols_py = columns_from_ptr(move(c_result))
    if left_index or right_index:
        index_ordered_dict = OrderedDict()
        index_ordered_dict[result_index_name] = all_cols_py.pop(result_index_pos)
        index_col = Table(index_ordered_dict)
    else:
        index_col = None

    data_ordered_dict = OrderedDict(zip(result_col_names,all_cols_py))

    return Table(data=data_ordered_dict, index=index_col)
