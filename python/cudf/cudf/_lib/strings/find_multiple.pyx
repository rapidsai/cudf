# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings.find_multiple cimport (
    find_multiple as cpp_find_multiple,
)

from cudf._lib.column cimport Column


@acquire_spill_lock()
def find_multiple(Column source_strings, Column target_strings):
    """
    Returns a column with character position values where each
    of the `target_strings` are found in each string of `source_strings`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef column_view target_view = target_strings.view()

    with nogil:
        c_result = move(cpp_find_multiple(
            source_view,
            target_view
        ))

    return Column.from_unique_ptr(move(c_result))
