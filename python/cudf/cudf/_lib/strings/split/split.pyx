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
from cudf._lib.scalar cimport DeviceScalar
from libcpp.string cimport string

from cudf._lib.cpp.strings.split.split cimport (
    split as cpp_split,
    rsplit as cpp_rsplit,
    split_record as cpp_split_record,
    rsplit_record as cpp_rsplit_record
)


def split(Column source_strings,
          object py_delimiter,
          size_type maxsplit):
    """
    Returns a Table by splitting the `source_strings`
    column around the specified `py_delimiter`.
    The split happens from beginning.
    """

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()
    cdef const string_scalar* scalar_str = <const string_scalar*>(
        delimiter.get_raw_ptr()
    )

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


def split_record(Column source_strings,
                 object py_delimiter,
                 size_type maxsplit):
    """
    Returns a Column by splitting the `source_strings`
    column around the specified `py_delimiter`.
    The split happens from beginning.
    """

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef const string_scalar* scalar_str = <const string_scalar*>(
        delimiter.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_split_record(
            source_view,
            scalar_str[0],
            maxsplit
        ))

    return Column.from_unique_ptr(
        move(c_result),
    )


def rsplit(Column source_strings,
           object py_delimiter,
           size_type maxsplit):
    """
    Returns a Table by splitting the `source_strings`
    column around the specified `py_delimiter`.
    The split happens from the end.
    """

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()
    cdef const string_scalar* scalar_str = <const string_scalar*>(
        delimiter.get_raw_ptr()
    )

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


def rsplit_record(Column source_strings,
                  object py_delimiter,
                  size_type maxsplit):
    """
    Returns a Column by splitting the `source_strings`
    column around the specified `py_delimiter`.
    The split happens from the end.
    """

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef const string_scalar* scalar_str = <const string_scalar*>(
        delimiter.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_rsplit_record(
            source_view,
            scalar_str[0],
            maxsplit
        ))

    return Column.from_unique_ptr(
        move(c_result),
    )
