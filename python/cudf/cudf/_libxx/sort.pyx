# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from cudf._libxx.move cimport move

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
from cudf._libxx.cpp.sort cimport (
    sorted_order, lower_bound, upper_bound, rank, rank_method
)
cimport cudf._libxx.cpp.types as libcudf_types


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

def rank_columns(Table source_table, str method, str na_option, bool ascending, bool pct=False):
    """
    Compute numerical data ranks (1 through n) of each column in the dataframe
    """
    cdef table_view source_table_view = source_table.view()
    
    cdef rank_method c_rank_method
    if  method == 'min':
        c_rank_method = rank_method.MIN
    elif  method == 'max':
        c_rank_method = rank_method.MAX
    elif  method == 'first':
        c_rank_method = rank_method.FIRST
    elif  method == 'dense':
        c_rank_method = rank_method.DENSE
    else:
        c_rank_method = rank_method.AVERAGE
    
    cdef libcudf_types.order column_order = (
        libcudf_types.order.ASCENDING
        if ascending
        else libcudf_types.order.DESCENDING
    )
    #ascending 
    #    #top    = na_is_smallest
    #    #bottom = na_is_largest
    #    #keep   = na_is_largest
    #descending
    #    #top    = na_is_largest
    #    #bottom = na_is_smallest
    #    #keep   = na_is_smallest
    cdef libcudf_types.null_order null_precedence
    if ascending:
        if na_option == 'top':
            null_precedence = libcudf_types.null_order.BEFORE
        else:
            null_precedence = libcudf_types.null_order.AFTER
    else:
        if na_option == 'top':
            null_precedence = libcudf_types.null_order.AFTER
        else:
            null_precedence = libcudf_types.null_order.BEFORE
    cdef libcudf_types.include_nulls _include_nulls = (
        libcudf_types.include_nulls.EXCLUDE_NULLS 
        if na_option == 'keep'
        else libcudf_types.include_nulls.INCLUDE_NULLS
    )
    cdef unique_ptr[table] c_result
 
    with nogil:
        c_result = move(
            rank(
                source_table_view,
                c_rank_method,
                column_order,
                _include_nulls,
                null_precedence
            )
        )

    return Table.from_unique_ptr(
        move(c_result), 
        column_names=source_table._column_names,
        index_names=(
            None if source_table._index is None 
            else source_table._index_names)
    )
