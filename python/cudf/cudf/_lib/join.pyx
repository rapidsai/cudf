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
from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
cimport cudf._lib.cpp.join as cpp_join


cpdef join(Table lhs, Table rhs, how=None):
    # left, inner and outer join
    cdef pair[unique_ptr[column], unique_ptr[column]] c_result
    cdef table_view c_lhs = lhs.view()
    cdef table_view c_rhs = rhs.view()

    if how == "inner":
        c_result = move(cpp_join.inner_join(
            c_lhs,
            c_rhs
        ))
    elif how == "left":
        c_result = move(cpp_join.left_join(
            c_lhs,
            c_rhs
        ))
    elif how == "outer":
        c_result = move(cpp_join.full_join(
            c_lhs,
            c_rhs
        ))
    else:
        raise ValueError(f"Invalid join type {how}")
    return (
        Column.from_unique_ptr(move(c_result.first)),
        Column.from_unique_ptr(move(c_result.second))
    )


cpdef semi_join(Table lhs, Table rhs, how=None):
    from cudf.core.column import as_column

    # left-semi and left-anti joins
    cdef unique_ptr[column] c_result
    cdef table_view c_lhs = lhs.view()
    cdef table_view c_rhs = rhs.view()

    if how == "leftsemi":
        c_result = move(cpp_join.left_semi_join(
            c_lhs,
            c_rhs
        ))
    elif how == "leftanti":
        c_result = move(cpp_join.left_anti_join(
            c_lhs,
            c_rhs
        ))
    else:
        raise ValueError(f"Invalid join type {how}")
    return Column.from_unique_ptr(move(c_result)), as_column([], dtype="int32")
