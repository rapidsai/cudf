# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.nvtext.edit_distance cimport (
    edit_distance as cpp_edit_distance,
    edit_distance_matrix as cpp_edit_distance_matrix,
)


@acquire_spill_lock()
def edit_distance(Column strings, Column targets):
    cdef column_view c_strings = strings.view()
    cdef column_view c_targets = targets.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_edit_distance(c_strings, c_targets))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def edit_distance_matrix(Column strings):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_edit_distance_matrix(c_strings))

    return Column.from_unique_ptr(move(c_result))
