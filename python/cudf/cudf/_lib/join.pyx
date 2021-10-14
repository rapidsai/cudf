# Copyright (c) 2020, NVIDIA CORPORATION.

from itertools import chain

import cudf

from libcpp cimport bool
from libcpp.memory cimport make_shared, make_unique, shared_ptr, unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

cimport cudf._lib.cpp.join as cpp_join
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport data_type, null_equality, size_type, type_id
from cudf._lib.utils cimport table_view_from_table

# The functions below return the *gathermaps* that represent
# the join result when joining on the keys `lhs` and `rhs`.

cdef class HashJoin:
    cdef unique_ptr[cpp_join.hash_join] join_obj

    def __cinit__(self, build_table):
        cdef table_view c_build_table = table_view_from_table(build_table)
        cdef null_equality c_null_equality = null_equality.EQUAL
        self.join_obj = make_unique[cpp_join.hash_join](
            c_build_table,
            c_null_equality
        )

    def join(self, probe_table, how):
        cdef pair[cpp_join.gather_map_type, cpp_join.gather_map_type] c_result

        if how == "inner":
            c_result = move(self.join_obj.get()[0].inner_join(
                table_view_from_table(probe_table)
            ))
        elif how == "left":
            c_result = move(self.join_obj.get()[0].left_join(
                table_view_from_table(probe_table)
            ))
        elif how == "outer":
            c_result = move(self.join_obj.get()[0].full_join(
                table_view_from_table(probe_table)
            ))
        else:
            raise ValueError(f"Invalid join type: {how}")

        cdef Column probe_rows = _gather_map_as_column(move(c_result.first))
        cdef Column build_rows = _gather_map_as_column(move(c_result.second))
        return probe_rows, build_rows


cpdef semi_join(lhs, rhs, how=None):
    # left-semi and left-anti joins
    cdef cpp_join.gather_map_type c_result
    cdef table_view c_lhs = table_view_from_table(lhs)
    cdef table_view c_rhs = table_view_from_table(rhs)

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

    cdef Column left_rows = _gather_map_as_column(move(c_result))
    return (
        left_rows,
        None
    )


cdef Column _gather_map_as_column(cpp_join.gather_map_type gather_map):
    # helple to convert a gather map to a Column
    cdef size_type size = gather_map.get()[0].size()
    cdef unique_ptr[column] c_col = make_unique[column](
        data_type(type_id.INT32),
        size,
        gather_map.get()[0].release())
    return Column.from_unique_ptr(move(c_col))
