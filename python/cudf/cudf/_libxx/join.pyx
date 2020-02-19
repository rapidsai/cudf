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

    if left_on is None:
        left_on = []
    if right_on is None:
        right_on = []

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
    print(lhs._data.keys())
    print(rhs._data.keys())
    for name in lhs._data.keys():
        result_col_names.append(name)
    for name in rhs._data.keys():
        if name not in lhs._data.keys():
            result_col_names.append(name)

    print('result_col_names:')
    print(result_col_names)

    result_index_pos = None
    # if left_index or right_index is true, then both indices must be processed as join columns
    if left_index or right_index:
        lhs_view = c_lhs.view()
        rhs_view = c_rhs.view()

        left_join_cols = [lhs.index.name] + list(lhs._data.keys())
        right_join_cols = [rhs.index.name] + list(rhs._data.keys())
        print('\n')
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
            print(result_col_names)
        elif right_index:
            right_on_idx = 0
            left_on_idx = left_join_cols.index(left_on[0])
            result_index_pos = 0
            result_index_name = lhs.index.name

        left_on_ind.push_back(left_on_idx)
        right_on_ind.push_back(right_on_idx)
        
        columns_in_common.push_back(pair[int,int]((left_on_idx, right_on_idx)))
        print('result_index_pos:')
        print(result_index_pos)
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

    print(left_on_ind)
    print(right_on_ind)
    print(columns_in_common)

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
    for i, x in enumerate(all_cols_py):
        print('column %s' % i)
        from cudf import Series
        print(Series(x))

    if left_index or right_index:
        index_ordered_dict = OrderedDict()
        index_ordered_dict[result_index_name] = all_cols_py.pop(result_index_pos)
        index_col = Table(index_ordered_dict)
    else:
        index_col = None

    data_ordered_dict = OrderedDict(zip(result_col_names,all_cols_py))

    return Table(data=data_ordered_dict, index=index_col)

    """
    for name, col in lhs._data.items():
        result_col_names.append(name)

    for name in left_on:
        left_on_ind.push_back(list(lhs._data.keys()).index(name) + lhs_offset)
        if (name in right_on and
           (left_on.index(name) == right_on.index(name))):
           # Index forms a hidden column
            columns_in_common.push_back(pair[int, int](
                list(lhs._data.keys()).index(name) + lhs_offset,
                list(rhs._data.keys()).index(name) + rhs_offset))

    for name in right_on:
        right_on_ind.push_back(list(rhs._data.keys()).index(name) + rhs_offset)

    for name, col in rhs._data.items():
        if not ((name in left_on) and (name in right_on)
        and (left_on.index(name) == right_on.index(name))):
            result_col_names.append(name)

    print('columns in common:')
    print(columns_in_common)
    """

    """
    # wont be entered if neither left index nor right index
    index_names = None
    result_index_name = None
    cdef int result_index_pos = 0

    # in pandas, whatever index is joined on, the opposite will
    # be returned in the result.
    if left_index or right_index:
        # both index columns will need to be joined
        lhs_view = c_lhs.view()
        rhs_view = c_rhs.view()
        lhs_offset += 1
        rhs_offset += 1
        if left_index:
            # include left index in 'on'
            left_on_ind.push_back(0)
            # the new index will be the index from the right, save its name
            result_index_name = rhs.index.name
            # the index we want to use in the future is the 0'th rhs column
            result_index_pos = len(lhs.columns) + 1
            # the common columns are the left index (0'th lhs) and index of the "right_on column"
            columns_in_common.push_back((0, list(rhs._data.keys()).index(right_on[0])+ 1))
        if right_index:
            # include right index in 'on'
            right_on_ind.push_back(0)
            # the new index will be the index from the left, save its name
            result_index_name = lhs.index.name
            # the index we want to use in the future is the 0'th lhs column
            result_index_pos = 0
            # the common columns are the "left_on column" and the index from the right (0'th rhs)
            columns_in_common.push_back((list(lhs._data.keys()).index(left_on[0])+ 1, 0))
    else:

        # if neither left_index nor right_index is specified,
        # we will always get a new RangeIndex
        lhs_view = c_lhs.data_view()
        rhs_view = c_rhs.data_view()

    # ok, so at this point we have two dicts, one that says the name and index of the index column,
    # and one that says the name and index of the common column in the index case. 
    result_col_names = []  # Preserve the order of the column names

    for name, col in lhs._data.items():
        result_col_names.append(name)

    for name in left_on:
        left_on_ind.push_back(list(lhs._data.keys()).index(name) + lhs_offset)
        if (name in right_on and
           (left_on.index(name) == right_on.index(name))):
           # Index forms a hidden column
            columns_in_common.push_back(pair[int, int](
                list(lhs._data.keys()).index(name) + lhs_offset,
                list(rhs._data.keys()).index(name) + rhs_offset))

    for name in right_on:
        right_on_ind.push_back(list(rhs._data.keys()).index(name) + rhs_offset)

    for name, col in rhs._data.items():
        if not ((name in left_on) and (name in right_on)
        and (left_on.index(name) == right_on.index(name))):
            result_col_names.append(name)

    print('columns in common:')
    print(columns_in_common)
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
    for i, x in enumerate(all_cols_py):
        print('column %s' % i)
        from cudf import Series
        print(Series(x))
    num_cols = len(all_cols_py)

    # get the index column
    index_ordered_dict = OrderedDict()
    index_ordered_dict[result_index_name] = all_cols_py[result_index_pos]
    index_col = Table(index_ordered_dict)

    data_ordered_dict = OrderedDict(zip(result_col_names, all_cols_py))

    return Table(data=data_ordered_dict, index=index_col)
    """