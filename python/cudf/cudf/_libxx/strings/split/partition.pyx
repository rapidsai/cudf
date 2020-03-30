# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.move cimport move
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.scalar.scalar cimport string_scalar
from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
from cudf._libxx.scalar cimport Scalar
from libcpp.string cimport string

from cudf._libxx.cpp.strings.split.partition cimport (
    partition as cpp_partition,
    rpartition as cpp_rpartition,
)


def partition(Column source_strings,
              Scalar delimiter):
    """
    Returns a Table by splitting the `source_strings`
    column at the first occurrence of the specified `delimiter`.
    """
    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()
    cdef string_scalar* scalar_str = <string_scalar*>(delimiter.c_value.get())

    with nogil:
        c_result = move(cpp_partition(
            source_view,
            scalar_str[0]
        ))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )


def rpartition(Column source_strings,
               Scalar delimiter):
    """
    Returns a Column by splitting the `source_strings`
    column at the last occurrence of the specified `delimiter`.
    """
    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()
    cdef string_scalar* scalar_str = <string_scalar*>(delimiter.c_value.get())

    with nogil:
        c_result = move(cpp_rpartition(
            source_view,
            scalar_str[0]
        ))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )
