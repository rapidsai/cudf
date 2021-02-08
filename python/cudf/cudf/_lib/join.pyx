# Copyright (c) 2020, NVIDIA CORPORATION.

from collections import OrderedDict
from itertools import chain

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table, columns_from_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
cimport cudf._lib.cpp.join as cpp_join


cpdef join(Table lhs, Table rhs, left_on, right_on, how=None):
    # left, inner and outer join
    cdef vector[int] c_left_on = left_on
    cdef vector[int] c_right_on = right_on
    cdef pair[unique_ptr[column], unique_ptr[column]] c_result
    cdef table_view c_lhs = lhs.view()
    cdef table_view c_rhs = rhs.view()

    if how == "inner":
        c_result = move(cpp_join.inner_join(
            c_lhs,
            c_rhs,
            c_left_on,
            c_right_on,
        ))
    elif how == "left":
        c_result = move(cpp_join.left_join(
            c_lhs,
            c_rhs,
            c_left_on,
            c_right_on,
        ))
    elif how == "outer":
        c_result = move(cpp_join.outer_join(
            c_lhs,
            c_rhs
            c_left_on,
            c_right_on
        ))
    else:
        raise ValueError(f"Unkown join type {how}")
    return (
        Column.from_unique_ptr(move(c_result.first)),
        Column.from_unique_ptr(move(c_result.second))
    )


cpdef join_semi_anti(Table lhs, Table rhs, left_on, right_on, how=None):
    # left-semi and left-anti joins
    cdef vector[int] c_left_on = left_on
    cdef vector[int] c_right_on = right_on
    cdef unique_ptr[column] c_result
    cdef table_view c_lhs = lhs.view()
    cdef table_view c_rhs = rhs.view()

    if how == "semi":
        c_result = move(cpp_join.left_semi_join(
            c_lhs,
            c_rhs,
            c_left_on,
            c_right_on
        ))
    elif how == "anti":
        c_result = move(cpp_join.left_anti_join(
            c_lhs,
            c_rhs,
            c_left_on,
            c_right_on
        ))
    else:
        raise ValueError(f"Invalid join type {how}")
