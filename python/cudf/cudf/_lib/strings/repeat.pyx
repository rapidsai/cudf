# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.strings cimport repeat as cpp_repeat
from cudf._lib.cpp.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar


def repeat(Column source_strings,
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
