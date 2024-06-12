# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.strings.convert.convert_integers cimport (
    is_integer as cpp_is_integer,
)


@acquire_spill_lock()
def is_integer(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that have integers.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_is_integer(
            source_view
        ))

    return Column.from_unique_ptr(move(c_result))
