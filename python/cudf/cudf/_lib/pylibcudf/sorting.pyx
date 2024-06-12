# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf cimport sorting as cpp_sorting
from cudf._lib.pylibcudf.libcudf.aggregation cimport rank_method
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport null_order, null_policy, order

from .column cimport Column
from .table cimport Table


cpdef Column sorted_order(Table source_table, list column_order, list null_precedence):
    """Computes the row indices required to sort the table.

    Parameters
    ----------
    source_table : Table
        The table to sort.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Column
        The row indices required to sort the table.
    """
    cdef unique_ptr[column] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.sorted_order(
                source_table.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Column.from_libcudf(move(c_result))


cpdef Column stable_sorted_order(
    Table source_table,
    list column_order,
    list null_precedence,
):
    """Computes the row indices required to sort the table,
    preserving order of equal elements.

    Parameters
    ----------
    source_table : Table
        The table to sort.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Column
        The row indices required to sort the table.
    """
    cdef unique_ptr[column] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.stable_sorted_order(
                source_table.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Column.from_libcudf(move(c_result))


cpdef Column rank(
    Column input_view,
    rank_method method,
    order column_order,
    null_policy null_handling,
    null_order null_precedence,
    bool percentage,
):
    """Computes the rank of each element in the column.

    Parameters
    ----------
    input_view : Column
        The column to rank.
    method : rank_method
        The method to use for ranking ties.
    column_order : order
        Whether the column should be sorted in ascending or descending order.
    null_handling : null_policy
        Whether or not nulls should be included in the ranking.
    null_precedence : null_order
        Whether nulls should come before or after non-nulls.
    percentage : bool
        Whether to return the rank as a percentage.

    Returns
    -------
    Column
        The rank of each element in the column.
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_sorting.rank(
                input_view.view(),
                method,
                column_order,
                null_handling,
                null_precedence,
                percentage,
            )
        )
    return Column.from_libcudf(move(c_result))


cpdef bool is_sorted(Table tbl, list column_order, list null_precedence):
    """Checks if the table is sorted.

    Parameters
    ----------
    tbl : Table
        The table to check.
    column_order : List[ColumnOrder]
        Whether each column is expected to be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls are expected before or after non-nulls.

    Returns
    -------
    bool
        Whether the table is sorted.
    """
    cdef bool c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.is_sorted(
                tbl.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return c_result


cpdef Table segmented_sort_by_key(
    Table values,
    Table keys,
    Column segment_offsets,
    list column_order,
    list null_precedence,
):
    """Sorts the table by key, within segments.

    Parameters
    ----------
    values : Table
        The table to sort.
    keys : Table
        The table to sort by.
    segment_offsets : Column
        The offsets of the segments.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.segmented_sort_by_key(
                values.view(),
                keys.view(),
                segment_offsets.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Table stable_segmented_sort_by_key(
    Table values,
    Table keys,
    Column segment_offsets,
    list column_order,
    list null_precedence,
):
    """Sorts the table by key preserving order of equal elements,
    within segments.

    Parameters
    ----------
    values : Table
        The table to sort.
    keys : Table
        The table to sort by.
    segment_offsets : Column
        The offsets of the segments.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.stable_segmented_sort_by_key(
                values.view(),
                keys.view(),
                segment_offsets.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Table sort_by_key(
    Table values,
    Table keys,
    list column_order,
    list null_precedence,
):
    """Sorts the table by key.

    Parameters
    ----------
    values : Table
        The table to sort.
    keys : Table
        The table to sort by.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.sort_by_key(
                values.view(),
                keys.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Table stable_sort_by_key(
    Table values,
    Table keys,
    list column_order,
    list null_precedence,
):
    """Sorts the table by key preserving order of equal elements.

    Parameters
    ----------
    values : Table
        The table to sort.
    keys : Table
        The table to sort by.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.stable_sort_by_key(
                values.view(),
                keys.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Table sort(Table source_table, list column_order, list null_precedence):
    """Sorts the table.

    Parameters
    ----------
    source_table : Table
        The table to sort.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.sort(
                source_table.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Table stable_sort(Table source_table, list column_order, list null_precedence):
    """Sorts the table preserving order of equal elements.

    Parameters
    ----------
    source_table : Table
        The table to sort.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.stable_sort(
                source_table.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))
