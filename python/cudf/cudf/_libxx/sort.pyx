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
cimport cudf._libxx.cpp.types as libcudf_types

from cudf._libxx.sort cimport (
    is_sorted as cpp_is_sorted,
)


def is_sorted(Table source_table, list col_order=[], list null_prec=[]):
    """
    Checks whether the rows of a `table` are sorted in lexicographical order.

    Parameters
    ----------
    source_table : table whose columns are to be checked for sort order
    col_order : boolean list of boolean values indicating expected
                sort order of each column size must be len(columns) or empty
                if empty, all columns expected sort order is set to ascending
                False (0) - ascending
                True (1)  - descending
    null_prec : list of boolean values indicating desired order of null compared
                to other elements for each column size must be len(columns) or
                empty if empty, null order is set to before
                False (0) - before
                True (1)  - after

    Returns
    -------
    returns : boolean
              true, if sorted as expected in col_order and null_prec
              false, otherwise
    """
    
    cdef vector[order] column_order
    cdef vector[null_order] null_precedence

    if (0 < len(col_order)) and (source_table._num_columns != len(col_order)):
        # TODO: raise an exception.
        pass
    elif (0 == len(col_order)):
        column_order = vector[order](len(col_order), order.ASCENDING)
    elif (source_table._num_columns == len(col_order)):
        column_order = vector[order](len(col_order), order.ASCENDING)
        for i in range(len(col_order)):
            if (col_order[i]):
                column_order[i] = order.DESCENDING

    if (0 < len(null_prec)) and (source_table._num_columns != len(col_order)):
        # TODO: raise an exception
        pass
    elif (0 == len(null_prec)):
        null_precedence = vector[null_order](len(null_prec), null_order.BEFORE)
    elif (source_table._num_columns == len(null_prec)):
        null_precedence = vector[null_order](len(null_prec), null_order.BEFORE)
        for i in range(len(null_prec)):
            if(null_prec[i]):
                null_precedence[i] = null_order.AFTER

    cdef bool c_result
    cdef table_view source_table_view = source_table.view()
    with nogil:
        c_result = cpp_is_sorted(source_table_view,
                                 column_order,
                                 null_precedence)

    return c_result


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
    cdef vector[libcudf_types.order] column_order
    column_order.reserve(len(ascending))
    cdef libcudf_types.null_order pred = (
        libcudf_types.null_order.BEFORE
        if na_position == 1
        else libcudf_types.null_order.AFTER
    )
    cdef vector[libcudf_types.null_order] null_precedence = (
        vector[libcudf_types.null_order](
            source_table._num_columns,
            pred
        )
    )

    for i in ascending:
        if i is True:
            column_order.push_back(libcudf_types.order.ASCENDING)
        else:
            column_order.push_back(libcudf_types.order.DESCENDING)

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
    cdef vector[libcudf_types.order] column_order = (
        vector[libcudf_types.order](
            bins_view.num_columns(),
            libcudf_types.order.ASCENDING
        )
    )
    cdef vector[libcudf_types.null_order] null_precedence = (
        vector[libcudf_types.null_order](
            bins_view.num_columns(),
            libcudf_types.null_order.BEFORE
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
