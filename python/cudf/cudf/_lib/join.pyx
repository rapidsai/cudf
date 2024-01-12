# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport make_unique, unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport device_buffer

cimport cudf._lib.cpp.join as cpp_join
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport data_type, size_type, type_id
from cudf._lib.utils cimport table_view_from_columns

# The functions below return the *gathermaps* that represent
# the join result when joining on the keys `lhs` and `rhs`.


@acquire_spill_lock()
def join(list lhs, list rhs, how=None):
    cdef pair[cpp_join.gather_map_type, cpp_join.gather_map_type] c_result
    cdef table_view c_lhs = table_view_from_columns(lhs)
    cdef table_view c_rhs = table_view_from_columns(rhs)

    if how == "inner":
        with nogil:
            c_result = move(cpp_join.inner_join(c_lhs, c_rhs))
    elif how == "left":
        with nogil:
            c_result = move(cpp_join.left_join(c_lhs, c_rhs))
    elif how == "outer":
        with nogil:
            c_result = move(cpp_join.full_join(c_lhs, c_rhs))
    else:
        raise ValueError(f"Invalid join type {how}")

    cdef Column left_rows = _gather_map_as_column(move(c_result.first))
    cdef Column right_rows = _gather_map_as_column(move(c_result.second))
    return left_rows, right_rows


@acquire_spill_lock()
def semi_join(list lhs, list rhs, how=None):
    # left-semi and left-anti joins
    cdef cpp_join.gather_map_type c_result
    cdef table_view c_lhs = table_view_from_columns(lhs)
    cdef table_view c_rhs = table_view_from_columns(rhs)

    if how == "leftsemi":
        with nogil:
            c_result = move(cpp_join.left_semi_join(c_lhs, c_rhs))
    elif how == "leftanti":
        with nogil:
            c_result = move(cpp_join.left_anti_join(c_lhs, c_rhs))
    else:
        raise ValueError(f"Invalid join type {how}")

    cdef Column left_rows = _gather_map_as_column(move(c_result))
    return left_rows, None


cdef Column _gather_map_as_column(cpp_join.gather_map_type gather_map):
    # help to convert a gather map to a Column
    cdef device_buffer c_empty
    cdef size_type size = gather_map.get()[0].size()
    cdef unique_ptr[column] c_col = move(make_unique[column](
        data_type(type_id.INT32),
        size,
        gather_map.get()[0].release(), move(c_empty), 0))
    return Column.from_unique_ptr(move(c_col))
