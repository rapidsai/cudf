from cudf._libxx.lib cimport *
from cudf._libxx.table cimport Table, TableColumns
from cudf._libxx.includes.join cimport *
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from collections import OrderedDict

cpdef join(lhs, rhs, left_on, right_on, how, method, left_index=False, right_index=False):
    """
    Call libcudf++ join for full outer, inner and left joins.
    Returns a list of tuples [(column, valid, name), ...]
    """

    cdef vector[int] left_on_ind
    cdef vector[int] right_on_ind
    cdef vector[pair[int, int]] columns_in_common

    # convert both left and right to c table objects
    cdef Table c_lhs = lhs
    cdef Table c_rhs = rhs

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

    result_index_pos = None
    # if left_index or right_index is true, then both indices must be processed as join columns
    if left_index or right_index:
        lhs_view = c_lhs.view()
        rhs_view = c_rhs.view()

        left_join_cols = [lhs.index.name] + list(lhs._data.keys())
        right_join_cols = [rhs.index.name] + list(rhs._data.keys())
        if left_index and right_index:
            left_on_idx = 0
            right_on_idx = 0
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
        # if neither left_index nor right_index is specified,
        # we will always get a new RangeIndex
        lhs_view = c_lhs.data_view()
        rhs_view = c_rhs.data_view()

        left_join_cols = list(lhs._data.keys())
        right_join_cols = list(rhs._data.keys())

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

    cdef TableColumns all_cols
    all_cols = Table.columns_from_ptr(move(c_result))
    all_cols_py = all_cols.columns

    if left_index or right_index:
        index_ordered_dict = OrderedDict()
        index_ordered_dict[result_index_name] = all_cols_py.pop(result_index_pos)
        index_col = Table(index_ordered_dict)
    else:
        index_col = None

    data_ordered_dict = OrderedDict(zip(result_col_names,all_cols_py))

    return Table(data=data_ordered_dict, index=index_col)