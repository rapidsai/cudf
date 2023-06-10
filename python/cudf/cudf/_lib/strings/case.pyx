# Copyright (c) 2018-2022, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.strings.case cimport (
    swapcase as cpp_swapcase,
    to_lower as cpp_to_lower,
    to_upper as cpp_to_upper,
)


@acquire_spill_lock()
def to_upper(Column source_strings):
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_to_upper(source_view))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def to_lower(Column source_strings):
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_to_lower(source_view))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def swapcase(Column source_strings):
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_swapcase(source_view))

    return Column.from_unique_ptr(move(c_result))
