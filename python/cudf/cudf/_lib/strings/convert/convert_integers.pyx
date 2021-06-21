# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column

from cudf._lib.cpp.strings.convert.convert_integers cimport (
    is_integer as cpp_is_integer,
)


def is_integer(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that have intergers.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_is_integer(
            source_view
        ))

    return Column.from_unique_ptr(move(c_result))
