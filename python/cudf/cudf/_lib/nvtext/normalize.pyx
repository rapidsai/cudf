# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.nvtext.normalize cimport (
    normalize_characters as cpp_normalize_characters,
    normalize_spaces as cpp_normalize_spaces,
)


@acquire_spill_lock()
def normalize_spaces(Column strings):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_normalize_spaces(c_strings))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def normalize_characters(Column strings, bool do_lower=True):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_normalize_characters(c_strings, do_lower))

    return Column.from_unique_ptr(move(c_result))
