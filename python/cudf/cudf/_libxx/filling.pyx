# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cudf._libxx.column cimport Column
from cudf._libxx.column cimport Column
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport (
    column_view,
    mutable_column_view
)
from cudf._libxx.cpp.scalar.scalar cimport scalar
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.move cimport move
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.table cimport Table

cimport cudf._libxx.cpp.filling as cpp_filling


def fill_in_place(Column destination, int begin, int end, Scalar value):
    cdef mutable_column_view c_destination = destination.mutable_view()
    cdef size_type c_begin = <size_type> begin
    cdef size_type c_end = <size_type> end
    cdef scalar* c_value = value.c_value.get()

    cpp_filling.fill_in_place(
        c_destination,
        c_begin,
        c_end,
        c_value[0]
    )


def fill(Column destination, int begin, int end, Scalar value):
    cdef column_view c_destination = destination.view()
    cdef size_type c_begin = <size_type> begin
    cdef size_type c_end = <size_type> end
    cdef scalar* c_value = value.c_value.get()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_filling.fill(
            c_destination,
            c_begin,
            c_end,
            c_value[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def repeat(Table input, object count, bool check_count=False):
    if isinstance(count, Column):
        return _repeat_via_column(input, count, check_count)

    if isinstance(count, Scalar):
        return _repeat_via_scalar(input, count)

    raise TypeError(
        "Expected `count` to be Column or Scalar but got {}"
        .format(type(count))
    )


def _repeat_via_column(Table input, Column count, bool check_count):
    cdef table_view c_input = input.view()
    cdef column_view c_count = count.view()
    cdef bool c_check_count = check_count
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_filling.repeat(
            c_input,
            c_count,
            c_check_count
        ))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=input._column_names,
        index_names=input._index_names
    )


def _repeat_via_scalar(Table input, Scalar count):
    cdef table_view c_input = input.view()
    cdef scalar* c_count = count.c_value.get()
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_filling.repeat(
            c_input,
            c_count[0]
        ))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=input._column_names,
        index_names=input._index_names
    )
