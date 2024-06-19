# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.strings.wrap cimport wrap as cpp_wrap
from cudf._lib.pylibcudf.libcudf.types cimport size_type


@acquire_spill_lock()
def wrap(Column source_strings,
         size_type width):
    """
    Returns a Column by wrapping long strings
    in the Column to be formatted in paragraphs
    with length less than a given `width`.
    """

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_wrap(
            source_view,
            width
        ))

    return Column.from_unique_ptr(move(c_result))
