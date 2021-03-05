# Copyright (c) 2020, NVIDIA CORPORATION.

from collections import OrderedDict
from itertools import chain

from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table, columns_from_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.types cimport size_type, data_type, type_id
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
cimport cudf._lib.cpp.join as cpp_join


# The functions below return the *gathermaps* that represent
# the join result when joining on the keys `lhs` and `rhs`.

cpdef join(Table lhs, Table rhs, how=None):

    cdef pair[cpp_join.gather_map_type, cpp_join.gather_map_type] c_result
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

    cdef size_type join_size = c_result.first.get()[0].size()
    cdef unique_ptr[column] left_rows = make_unique[column](
        data_type(type_id.INT32),
        join_size,
        c_result.first.get()[0].release()
    )
    cdef unique_ptr[column] right_rows = make_unique[column](
        data_type(type_id.INT32),
        join_size,
        c_result.second.get()[0].release()
    )
    return (
        Column.from_unique_ptr(move(left_rows)),
        Column.from_unique_ptr(move(right_rows))
    )


cpdef semi_join(Table lhs, Table rhs, how=None):

    from cudf.core.column import as_column

    # left-semi and left-anti joins
    cdef cpp_join.gather_map_type c_result
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

    cdef size_type join_size = c_result.get()[0].size()
    cdef unique_ptr[column] left_rows = make_unique[column](
        data_type(type_id.INT32),
        join_size,
        c_result.get()[0].release()
    )
    return (
        Column.from_unique_ptr(move(left_rows)),
        as_column([], dtype="int32")
    )
