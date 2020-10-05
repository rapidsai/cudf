# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport Column
from cudf._lib.table cimport Table

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.scalar cimport Scalar
from libcpp.string cimport string

from cudf._lib.cpp.strings.split.split cimport (
    split as cpp_split,
    rsplit as cpp_rsplit,
)


def split(Column source_strings,
          Scalar delimiter,
          size_type maxsplit):
    """
    Returns a Table by splitting the `source_strings`
    column around the specified `delimiter`.
    The split happens from beginning.
    """
    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()
    cdef string_scalar* scalar_str = <string_scalar*>(delimiter.c_value.get())

    with nogil:
        c_result = move(cpp_split(
            source_view,
            scalar_str[0],
            maxsplit
        ))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )


def rsplit(Column source_strings,
           Scalar delimiter,
           size_type maxsplit):
    """
    Returns a Table by splitting the `source_strings`
    column around the specified `delimiter`.
    The split happens from the end.
    """
    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()
    cdef string_scalar* scalar_str = <string_scalar*>(delimiter.c_value.get())

    with nogil:
        c_result = move(cpp_rsplit(
            source_view,
            scalar_str[0],
            maxsplit
        ))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )
