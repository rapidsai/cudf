# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd

from cudf._libxx.column cimport *
from cudf._libxx.table cimport *
from cudf._libxx.lib cimport *
from cudf._libxx.move cimport move

from cudf._libxx.includes.column.column cimport column
from cudf._libxx.includes.table.table_view cimport table_view
from cudf._libxx.includes.sort cimport *


def order_by(Table source_table, object ascending, bool na_position):
    """
    Sorting the table ascending/descending

    Parameters
    ----------
    source_table : table which will be sorted
    ascending : list of boolean values which correspond to each column
                in source_table signifying order of each column
                True - Ascending and False - Descending
    na_position : whether null should be considered larget or smallest value
                  0 - largest and 1 - smallest

    """

    cdef table_view source_table_view = source_table.data_view()
    cdef vector[order] column_order
    column_order.reserve(len(ascending))
    cdef null_order pred = (
        null_order.BEFORE if na_position == 1 else null_order.AFTER)
    cdef vector[null_order] null_precedence = vector[null_order](
        source_table._num_columns, pred)

    for i in ascending:
        if i is True:
            column_order.push_back(order.ASCENDING)
        else:
            column_order.push_back(order.DESCENDING)

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(sorted_order(source_table_view,
                                     column_order,
                                     null_precedence))

    return Column.from_unique_ptr(move(c_result))


def digitize(Table source_values_table, Table bins, bool right=False):
    """
    Return the indices of the bins to which each value in source_table belongs.

    Parameters
    ----------
    source_table : Input table to be binned.
    bins : Table containing columns of bins
    right : Indicating whether the intervals include the
            right or the left bin edge.
    """

    cdef table_view bins_view = bins.view()
    cdef table_view source_values_table_view = source_values_table.view()
    cdef vector[order] column_order = vector[order](bins_view.num_columns(),
                                                    order.ASCENDING)
    cdef vector[null_order] null_precedence = vector[null_order](
        bins_view.num_columns(),
        null_order.BEFORE)

    cdef unique_ptr[column] c_result
    if right is True:
        with nogil:
            c_result = move(lower_bound(
                bins_view,
                source_values_table_view,
                column_order,
                null_precedence)
            )
    else:
        with nogil:
            c_result = move(upper_bound(
                bins_view,
                source_values_table_view,
                column_order,
                null_precedence)
            )

    return Column.from_unique_ptr(move(c_result))
