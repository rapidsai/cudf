# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from libcpp.vector cimport vector
cimport cudf._libxx.includes.search as cpp_search


def search_sorted(
    Table table, Table values, side, ascending=True, na_position="last"
):
    """Find indices where elements should be inserted to maintain order

    Parameters
    ----------
    table : Table
        Table to search in
    values : Table
        Table of values to search for
    side : str {‘left’, ‘right’} optional
        If ‘left’, the index of the first suitable location is given.
        If ‘right’, return the last such index
    """
    cdef unique_ptr[column] c_result
    cdef vector[order] c_column_order
    cdef vector[null_order] c_null_precedence
    cdef order c_order
    cdef null_order c_null_order
    cdef table_view c_table_data = table.data_view()
    cdef table_view c_values_data = values.data_view()

    # Note: We are ignoring index columns here
    c_order = order.ASCENDING if ascending else order.DESCENDING
    c_null_order = (
        null_order.AFTER if na_position=="last" else null_order.BEFORE
    )
    c_column_order = vector[order](table._num_columns, c_order)
    c_null_precedence = vector[null_order](table._num_columns, c_null_order)

    if side == 'left':
        with nogil:
            c_result = move(
                cpp_search.lower_bound(
                    c_table_data,
                    c_values_data,
                    c_column_order,
                    c_null_precedence,
                )
            )
    elif side == 'right':
        with nogil:
            c_result = move(
                cpp_search.upper_bound(
                    c_table_data,
                    c_values_data,
                    c_column_order,
                    c_null_precedence,
                )
            )
    return Column.from_unique_ptr(move(c_result))


def contains(Column haystack, Column needles):
    """Check whether column contains multiple values

    Parameters
    ----------
    column : NumericalColumn
        Column to search in
    needles :
        A column of values to search for
    """
    cdef unique_ptr[column] c_result
    cdef column_view c_haystack = haystack.view()
    cdef column_view c_needles = needles.view()

    with nogil:
        c_result = move(
            cpp_search.contains(
                c_haystack,
                c_needles,
            )
        )
    return Column.from_unique_ptr(move(c_result))
