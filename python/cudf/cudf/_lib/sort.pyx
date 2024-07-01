# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from itertools import repeat

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.aggregation cimport rank_method
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.search cimport lower_bound, upper_bound
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport null_order, order as cpp_order
from cudf._lib.utils cimport (
    columns_from_pylibcudf_table,
    table_view_from_columns,
)

from cudf._lib import pylibcudf


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

    if ascending is None:
        column_order = [pylibcudf.types.Order.ASCENDING] * len(source_columns)
    else:
        if len(ascending) != len(source_columns):
            raise ValueError(
                f"Expected a list-like of length {len(source_columns)}, "
                f"got length {len(ascending)} for `ascending`"
            )
        column_order = [pylibcudf.types.Order.DESCENDING] * len(source_columns)
        for idx, val in enumerate(ascending):
            if val:
                column_order[idx] = pylibcudf.types.Order.ASCENDING

    if null_position is None:
        null_precedence = [pylibcudf.types.NullOrder.AFTER] * len(source_columns)
    else:
        if len(null_position) != len(source_columns):
            raise ValueError(
                f"Expected a list-like of length {len(source_columns)}, "
                f"got length {len(null_position)} for `null_position`"
            )
        null_precedence = [pylibcudf.types.NullOrder.AFTER] * len(source_columns)
        for idx, val in enumerate(null_position):
            if val:
                null_precedence[idx] = pylibcudf.types.NullOrder.BEFORE

    return pylibcudf.sorting.is_sorted(
        pylibcudf.Table(
            [c.to_pylibcudf(mode="read") for c in source_columns]
        ),
        column_order,
        null_precedence
    )


def ordering(column_order, null_precedence):
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
    c_column_order = []
    c_null_precedence = []
    for asc, null in zip(column_order, null_precedence):
        c_column_order.append(
            pylibcudf.types.Order.ASCENDING if asc else pylibcudf.types.Order.DESCENDING
        )
        if asc ^ (null == "first"):
            c_null_precedence.append(pylibcudf.types.NullOrder.AFTER)
        elif asc ^ (null == "last"):
            c_null_precedence.append(pylibcudf.types.NullOrder.BEFORE)
        else:
            raise ValueError(f"Invalid null precedence {null}")
    return c_column_order, c_null_precedence


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
    order = ordering(ascending, repeat(na_position))
    func = getattr(pylibcudf.sorting, f"{'stable_' if stable else ''}sorted_order")

    return Column.from_pylibcudf(
        func(
            pylibcudf.Table(
                [c.to_pylibcudf(mode="read") for c in columns_from_table],
            ),
            order[0],
            order[1],
        )
    )


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
    ncol = len(values)
    order = ordering(
        column_order or repeat(True, ncol),
        null_precedence or repeat("first", ncol),
    )
    return columns_from_pylibcudf_table(
        pylibcudf.sorting.sort(
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in values]),
            order[0],
            order[1],
        )
    )


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
    order = ordering(ascending, na_position)
    func = getattr(pylibcudf.sorting, f"{'stable_' if stable else ''}sort_by_key")
    return columns_from_pylibcudf_table(
        func(
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in values]),
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in keys]),
            order[0],
            order[1],
        )
    )


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
    ncol = len(values)
    order = ordering(
        column_order or repeat(True, ncol),
        null_precedence or repeat("first", ncol),
    )
    func = getattr(
        pylibcudf.sorting,
        f"{'stable_' if stable else ''}segmented_sort_by_key"
    )
    return columns_from_pylibcudf_table(
        func(
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in values]),
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in keys]),
            segment_offsets.to_pylibcudf(mode="read"),
            order[0],
            order[1],
        )
    )


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
    cdef vector[cpp_order] column_order = (
        vector[cpp_order](
            bins_view.num_columns(),
            cpp_order.ASCENDING
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
def rank_columns(list source_columns, rank_method method, str na_option,
                 bool ascending, bool pct
                 ):
    """
    Compute numerical data ranks (1 through n) of each column in the dataframe
    """
    column_order = (
        pylibcudf.types.Order.ASCENDING
        if ascending
        else pylibcudf.types.Order.DESCENDING
    )
    # ascending
    #    #top    = na_is_smallest
    #    #bottom = na_is_largest
    #    #keep   = na_is_largest
    # descending
    #    #top    = na_is_largest
    #    #bottom = na_is_smallest
    #    #keep   = na_is_smallest
    if ascending:
        if na_option == 'top':
            null_precedence = pylibcudf.types.NullOrder.BEFORE
        else:
            null_precedence = pylibcudf.types.NullOrder.AFTER
    else:
        if na_option == 'top':
            null_precedence = pylibcudf.types.NullOrder.AFTER
        else:
            null_precedence = pylibcudf.types.NullOrder.BEFORE
    c_null_handling = (
        pylibcudf.types.NullPolicy.EXCLUDE
        if na_option == 'keep'
        else pylibcudf.types.NullPolicy.INCLUDE
    )

    return [
        Column.from_pylibcudf(
            pylibcudf.sorting.rank(
                col.to_pylibcudf(mode="read"),
                method,
                column_order,
                c_null_handling,
                null_precedence,
                pct,
            )
        )
        for col in source_columns
    ]
