# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.strings cimport repeat as cpp_repeat
from cudf._lib.pylibcudf.libcudf.types cimport size_type


@acquire_spill_lock()
def repeat_scalar(Column source_strings,
                  size_type repeats):
    """
    Returns a Column after repeating
    each string in `source_strings`
    `repeats` number of times.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_repeat.repeat_strings(
            source_view,
            repeats
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def repeat_sequence(Column source_strings,
                    Column repeats):
    """
    Returns a Column after repeating
    each string in `source_strings`
    `repeats` number of times.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef column_view repeats_view = repeats.view()

    with nogil:
        c_result = move(cpp_repeat.repeat_strings(
            source_view,
            repeats_view
        ))

    return Column.from_unique_ptr(move(c_result))
