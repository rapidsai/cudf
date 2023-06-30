# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from itertools import repeat

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move, pair
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.cpp.aggregation cimport (
    rank_method,
    underlying_type_t_rank_method,
)
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.search cimport lower_bound, upper_bound
from cudf._lib.cpp.sorting cimport (
    is_sorted as cpp_is_sorted,
    rank,
    segmented_sort_by_key as cpp_segmented_sort_by_key,
    sort as cpp_sort,
    sort_by_key as cpp_sort_by_key,
    sorted_order,
    stable_segmented_sort_by_key as cpp_stable_segmented_sort_by_key,
    stable_sort_by_key as cpp_stable_sort_by_key,
    stable_sorted_order,
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport null_order, null_policy, order
from cudf._lib.utils cimport columns_from_unique_ptr, table_view_from_columns


@acquire_spill_lock()
def is_sorted(
    list source_columns, object ascending=None, object null_position=None
):
    """
    Checks whether the rows of a `table` are sorted in lexicographical order.

    Parameters
    ----------
    source_columns : list of columns
        columns to be checked for sort order
    ascending : None or list-like of booleans
        None or list-like of boolean values indicating expected sort order of
        each column. If list-like, size of list-like must be len(columns). If
        None, all columns expected sort order is set to ascending. False (0) -
        descending, True (1) - ascending.
    null_position : None or list-like of booleans
        None or list-like of boolean values indicating desired order of nulls
        compared to other elements. If list-like, size of list-like must be
        len(columns). If None, null order is set to before. False (0) - after,
        True (1) - before.

    Returns
    -------
    returns : boolean
        Returns True, if sorted as expected by ``ascending`` and
        ``null_position``, False otherwise.
    """

    cdef vector[order] column_order
    cdef vector[null_order] null_precedence

    if ascending is None:
        column_order = vector[order](len(source_columns), order.ASCENDING)
    else:
        if len(ascending) != len(source_columns):
            raise ValueError(
                f"Expected a list-like of length {len(source_columns)}, "
                f"got length {len(ascending)} for `ascending`"
            )
        column_order = vector[order](
            len(source_columns), order.DESCENDING
        )
        for idx, val in enumerate(ascending):
            if val:
                column_order[idx] = order.ASCENDING

    if null_position is None:
        null_precedence = vector[null_order](
            len(source_columns), null_order.AFTER
        )
    else:
        if len(null_position) != len(source_columns):
            raise ValueError(
                f"Expected a list-like of length {len(source_columns)}, "
                f"got length {len(null_position)} for `null_position`"
            )
        null_precedence = vector[null_order](
            len(source_columns), null_order.AFTER
        )
        for idx, val in enumerate(null_position):
            if val:
                null_precedence[idx] = null_order.BEFORE

    cdef bool c_result
    cdef table_view source_table_view = table_view_from_columns(source_columns)
    with nogil:
        c_result = cpp_is_sorted(
            source_table_view,
            column_order,
            null_precedence
        )

    return c_result


cdef pair[vector[order], vector[null_order]] ordering(
    column_order, null_precedence
):
    """
    Construct order and null order vectors

    Parameters
    ----------
    column_order
        Iterable of bool (True for ascending order, False for descending)
    null_precedence
        Iterable string for null positions ("first" for start, "last" for end)

    Both iterables must be the same length (not checked)

    Returns
    -------
    pair of vectors (order, and null_order)
    """
    cdef vector[order] c_column_order
    cdef vector[null_order] c_null_precedence
    for asc, null in zip(column_order, null_precedence):
        c_column_order.push_back(order.ASCENDING if asc else order.DESCENDING)
        if asc ^ (null == "first"):
            c_null_precedence.push_back(null_order.AFTER)
        elif asc ^ (null == "last"):
            c_null_precedence.push_back(null_order.BEFORE)
        else:
            raise ValueError(f"Invalid null precedence {null}")
    return pair[vector[order], vector[null_order]](
        c_column_order, c_null_precedence
    )


@acquire_spill_lock()
def order_by(
    list columns_from_table,
    object ascending,
    str na_position,
    *,
    bool stable
):
    """
    Get index to sort the table in ascending/descending order.

    Parameters
    ----------
    columns_from_table : list[Column]
        Columns from the table which will be sorted
    ascending : sequence[bool]
         Sequence of boolean values which correspond to each column
         in the table to be sorted signifying the order of each column
         True - Ascending and False - Descending
    na_position : str
        Whether null values should show up at the "first" or "last"
        position of **all** sorted column.
    stable : bool
        Should the sort be stable? (no default)

    Returns
    -------
    Column of indices that sorts the table
    """
    cdef table_view source_table_view = table_view_from_columns(
        columns_from_table
    )
    cdef pair[vector[order], vector[null_order]] order = ordering(
        ascending, repeat(na_position)
    )
    cdef unique_ptr[column] c_result
    if stable:
        with nogil:
            c_result = move(stable_sorted_order(source_table_view,
                                                order.first,
                                                order.second))
    else:
        with nogil:
            c_result = move(sorted_order(source_table_view,
                                         order.first,
                                         order.second))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def sort(
    list values,
    list column_order=None,
    list null_precedence=None,
):
    """
    Sort the table in ascending/descending order.

    Parameters
    ----------
    values : list[Column]
        Columns of the table which will be sorted
    column_order : list[bool], optional
        Sequence of boolean values which correspond to each column in
        keys providing the sort order (default all True).
        With True <=> ascending; False <=> descending.
    null_precedence : list[str], optional
        Sequence of "first" or "last" values (default "first")
        indicating the position of null values when sorting the keys.
    """
    cdef table_view values_view = table_view_from_columns(values)
    cdef unique_ptr[table] result
    ncol = len(values)
    cdef pair[vector[order], vector[null_order]] order = ordering(
        column_order or repeat(True, ncol),
        null_precedence or repeat("first", ncol),
    )
    with nogil:
        result = move(
            cpp_sort(
                values_view,
                order.first,
                order.second,
            )
        )
    return columns_from_unique_ptr(move(result))


@acquire_spill_lock()
def sort_by_key(
    list values,
    list keys,
    object ascending,
    object na_position,
    *,
    bool stable,
):
    """
    Sort a table by given keys

    Parameters
    ----------
    values : list[Column]
        Columns of the table which will be sorted
    keys : list[Column]
        Columns making up the sort key
    ascending : list[bool]
        Sequence of boolean values which correspond to each column
        in the table to be sorted signifying the order of each column
        True - Ascending and False - Descending
    na_position : list[str]
        Sequence of "first" or "last" values (default "first")
        indicating the position of null values when sorting the keys.
    stable : bool
        Should the sort be stable? (no default)

    Returns
    -------
    list[Column]
        list of value columns sorted by keys
    """
    cdef table_view value_view = table_view_from_columns(values)
    cdef table_view key_view = table_view_from_columns(keys)
    cdef pair[vector[order], vector[null_order]] order = ordering(
        ascending, na_position
    )
    cdef unique_ptr[table] c_result
    if stable:
        with nogil:
            c_result = move(cpp_stable_sort_by_key(value_view,
                                                   key_view,
                                                   order.first,
                                                   order.second))
    else:
        with nogil:
            c_result = move(cpp_sort_by_key(value_view,
                                            key_view,
                                            order.first,
                                            order.second))

    return columns_from_unique_ptr(move(c_result))


@acquire_spill_lock()
def segmented_sort_by_key(
    list values,
    list keys,
    Column segment_offsets,
    list column_order=None,
    list null_precedence=None,
    *,
    bool stable,
):
    """
    Sort segments of a table by given keys

    Parameters
    ----------
    values : list[Column]
        Columns of the table which will be sorted
    keys : list[Column]
        Columns making up the sort key
    offsets : Column
        Segment offsets
    column_order : list[bool], optional
        Sequence of boolean values which correspond to each column in
        keys providing the sort order (default all True).
        With True <=> ascending; False <=> descending.
    null_precedence : list[str], optional
        Sequence of "first" or "last" values (default "first")
        indicating the position of null values when sorting the keys.
    stable : bool
        Should the sort be stable? (no default)

    Returns
    -------
    list[Column]
        list of value columns sorted by keys
    """
    cdef table_view values_view = table_view_from_columns(values)
    cdef table_view keys_view = table_view_from_columns(keys)
    cdef column_view offsets_view = segment_offsets.view()
    cdef unique_ptr[table] result
    ncol = len(values)
    cdef pair[vector[order], vector[null_order]] order = ordering(
        column_order or repeat(True, ncol),
        null_precedence or repeat("first", ncol),
    )
    if stable:
        with nogil:
            result = move(
                cpp_stable_segmented_sort_by_key(
                    values_view,
                    keys_view,
                    offsets_view,
                    order.first,
                    order.second,
                )
            )
    else:
        with nogil:
            result = move(
                cpp_segmented_sort_by_key(
                    values_view,
                    keys_view,
                    offsets_view,
                    order.first,
                    order.second,
                )
            )
    return columns_from_unique_ptr(move(result))


@acquire_spill_lock()
def digitize(list source_columns, list bins, bool right=False):
    """
    Return the indices of the bins to which each value in source_table belongs.

    Parameters
    ----------
    source_columns : Input columns to be binned.
    bins : List containing columns of bins
    right : Indicating whether the intervals include the
            right or the left bin edge.
    """

    cdef table_view bins_view = table_view_from_columns(bins)
    cdef table_view source_table_view = table_view_from_columns(
        source_columns
    )
    cdef vector[order] column_order = (
        vector[order](
            bins_view.num_columns(),
            order.ASCENDING
        )
    )
    cdef vector[null_order] null_precedence = (
        vector[null_order](
            bins_view.num_columns(),
            null_order.BEFORE
        )
    )

    cdef unique_ptr[column] c_result
    if right:
        with nogil:
            c_result = move(lower_bound(
                bins_view,
                source_table_view,
                column_order,
                null_precedence)
            )
    else:
        with nogil:
            c_result = move(upper_bound(
                bins_view,
                source_table_view,
                column_order,
                null_precedence)
            )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def rank_columns(list source_columns, object method, str na_option,
                 bool ascending, bool pct
                 ):
    """
    Compute numerical data ranks (1 through n) of each column in the dataframe
    """
    cdef rank_method c_rank_method = < rank_method > (
        < underlying_type_t_rank_method > method
    )

    cdef order column_order = (
        order.ASCENDING
        if ascending
        else order.DESCENDING
    )
    # ascending
    #    #top    = na_is_smallest
    #    #bottom = na_is_largest
    #    #keep   = na_is_largest
    # descending
    #    #top    = na_is_largest
    #    #bottom = na_is_smallest
    #    #keep   = na_is_smallest
    cdef null_order null_precedence
    if ascending:
        if na_option == 'top':
            null_precedence = null_order.BEFORE
        else:
            null_precedence = null_order.AFTER
    else:
        if na_option == 'top':
            null_precedence = null_order.AFTER
        else:
            null_precedence = null_order.BEFORE
    cdef null_policy c_null_handling = (
        null_policy.EXCLUDE
        if na_option == 'keep'
        else null_policy.INCLUDE
    )
    cdef bool percentage = pct

    cdef vector[unique_ptr[column]] c_results
    cdef column_view c_view
    cdef Column col
    for col in source_columns:
        c_view = col.view()
        with nogil:
            c_results.push_back(move(
                rank(
                    c_view,
                    c_rank_method,
                    column_order,
                    c_null_handling,
                    null_precedence,
                    percentage
                )
            ))

    return [Column.from_unique_ptr(
        move(c_results[i])
    ) for i in range(c_results.size())]
