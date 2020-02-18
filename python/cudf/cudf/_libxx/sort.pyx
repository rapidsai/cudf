# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from __future__ import print_function
import pandas as pd

from cudf._libxx.column cimport *
from cudf._libxx.table cimport *
from cudf._libxx.lib cimport *

from cudf._libxx.includes.sort cimport *


def order_by(Table source_table, ascending, na_position):
    cdef table_view source_table_view  = source_table.data_view()
    # Adding the first value which will meant for index column
    # which would be added when view is created
    #cdef vector[order] column_order = vector[order](source_table_view.num_columns() - source_table._num_columns , order.ASCENDING)
    cdef vector[order] column_order
    cdef vector[null_order] null_precedence
    for i in ascending:
        if i is True:
            column_order.push_back(order.ASCENDING)
        else:
            column_order.push_back(order.DESCENDING)
    #print ("RGSL : Size of order vector", column_order.size()) 
    #print ("RGSL : number of columns", source_table._num_columns) 
    #print ("RGSL : number of columns in table view", source_table_view.num_columns()) 
    cdef null_order pred = null_order.BEFORE if na_position == 1 else null_order.AFTER
    print ("RGSL : The null order ", na_position)
    
    for i in range(source_table._num_columns):
        null_precedence.push_back(pred)

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(sorted_order(source_table_view,
                                     column_order,
                                     null_precedence))

    return Column.from_unique_ptr(move(c_result))

def digitize(Table source_values_table, Table bins, right=False):
    cdef table_view bins_view = bins.view()
    cdef table_view source_values_table_view = source_values_table.view()
    cdef vector[order] column_order
    cdef vector[null_order] null_precedence
    for i in range(bins_view.num_columns()):
        column_order.push_back(order.ASCENDING)
        null_precedence.push_back(null_order.BEFORE)

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

    

 
    
