# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from cudf._libxx.move cimport move

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table_view cimport table_view
from cudf._libxx.cpp.sort cimport (
    sorted_order, lower_bound, upper_bound
)
cimport cudf._libxx.cpp.types as cudf_types


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
    cdef vector[cudf_types.order] column_order
    column_order.reserve(len(ascending))
    cdef cudf_types.null_order pred = (
        cudf_types.null_order.BEFORE
        if na_position == 1
        else cudf_types.null_order.AFTER
    )
    cdef vector[cudf_types.null_order] null_precedence = (
        vector[cudf_types.null_order](
            source_table._num_columns,
            pred
        )
    )

    for i in ascending:
        if i is True:
            column_order.push_back(cudf_types.order.ASCENDING)
        else:
            column_order.push_back(cudf_types.order.DESCENDING)

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
    cdef vector[cudf_types.order] column_order = vector[cudf_types.order](
        bins_view.num_columns(),
        cudf_types.order.ASCENDING
    )
    cdef vector[cudf_types.null_order] null_precedence = (
        vector[cudf_types.null_order](
            bins_view.num_columns(),
            cudf_types.null_order.BEFORE
        )
    )

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
