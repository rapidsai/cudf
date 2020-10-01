# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport (
    column_view,
    mutable_column_view
)
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.scalar cimport Scalar
from cudf._lib.table cimport Table

cimport cudf._lib.cpp.filling as cpp_filling


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


def repeat(Table inp, object count, bool check_count=False):
    if isinstance(count, Column):
        return _repeat_via_column(inp, count, check_count)
    else:
        return _repeat_via_size_type(inp, count)


def _repeat_via_column(Table inp, Column count, bool check_count):
    cdef table_view c_inp = inp.view()
    cdef column_view c_count = count.view()
    cdef bool c_check_count = check_count
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_filling.repeat(
            c_inp,
            c_count,
            c_check_count
        ))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=inp._column_names,
        index_names=inp._index_names
    )


def _repeat_via_size_type(Table inp, size_type count):
    cdef table_view c_inp = inp.view()
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_filling.repeat(
            c_inp,
            count
        ))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=inp._column_names,
        index_names=inp._index_names
    )


def sequence(int size, Scalar init, Scalar step):
    cdef size_type c_size = size
    cdef scalar* c_init = init.c_value.get()
    cdef scalar* c_step = step.c_value.get()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_filling.sequence(
            c_size,
            c_init[0],
            c_step[0]
        ))

    return Column.from_unique_ptr(move(c_result))
