# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.move cimport move
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.table.table_view cimport table_view
from cudf._libxx.cpp.scalar.scalar cimport string_scalar
from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.column cimport Column
from libcpp.memory cimport unique_ptr
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.scalar cimport Scalar
from libcpp.string cimport string
from cudf._libxx.table cimport Table

from cudf._libxx.cpp.strings.combine cimport (
    concatenate as cpp_concatenate,
    join_strings as cpp_join_strings
)


def concatenate(Table source_strings,
                Scalar separator,
                Scalar narep):
    """
    Returns a Column by concatenating strings column-wise in `source_strings`
    with the specified `separator` between each column and
    `na`/`None` values are replaced by `narep`
    """
    cdef unique_ptr[column] c_result
    cdef table_view source_view = source_strings.data_view()

    cdef string_scalar* scalar_separator = \
        <string_scalar*>(separator.c_value.get())
    cdef string_scalar* scalar_narep = <string_scalar*>(narep.c_value.get())

    with nogil:
        c_result = move(cpp_concatenate(
            source_view,
            scalar_separator[0],
            scalar_narep[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def join(Column source_strings,
         Scalar separator,
         Scalar narep):
    """
    Returns a Column by concatenating strings row-wise in `source_strings`
    with the specified `separator` between each column and
    `na`/`None` values are replaced by `narep`
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_separator = \
        <string_scalar*>(separator.c_value.get())
    cdef string_scalar* scalar_narep = <string_scalar*>(narep.c_value.get())

    with nogil:
        c_result = move(cpp_join_strings(
            source_view,
            scalar_separator[0],
            scalar_narep[0]
        ))

    return Column.from_unique_ptr(move(c_result))
