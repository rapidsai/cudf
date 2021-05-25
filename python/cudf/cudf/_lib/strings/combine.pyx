# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport Column
from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.column.column cimport column
from cudf._lib.scalar cimport DeviceScalar
from libcpp.string cimport string
from cudf._lib.table cimport Table

from cudf._lib.cpp.strings.combine cimport (
    concatenate as cpp_concatenate,
    join_strings as cpp_join_strings,
    join_list_elements as cpp_join_list_elements
)


def concatenate(Table source_strings,
                object py_separator,
                object py_narep):
    """
    Returns a Column by concatenating strings column-wise in `source_strings`
    with the specified `py_separator` between each column and
    `na`/`None` values are replaced by `py_narep`
    """
    cdef DeviceScalar separator = py_separator.device_value
    cdef DeviceScalar narep = py_narep.device_value

    cdef unique_ptr[column] c_result
    cdef table_view source_view = source_strings.data_view()

    cdef const string_scalar* scalar_separator = \
        <const string_scalar*>(separator.get_raw_ptr())
    cdef const string_scalar* scalar_narep = <const string_scalar*>(
        narep.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_concatenate(
            source_view,
            scalar_separator[0],
            scalar_narep[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def join(Column source_strings,
         object py_separator,
         object py_narep):
    """
    Returns a Column by concatenating strings row-wise in `source_strings`
    with the specified `py_separator` between each column and
    `na`/`None` values are replaced by `py_narep`
    """

    cdef DeviceScalar separator = py_separator.device_value
    cdef DeviceScalar narep = py_narep.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_separator = \
        <const string_scalar*>(separator.get_raw_ptr())
    cdef const string_scalar* scalar_narep = <const string_scalar*>(
        narep.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_join_strings(
            source_view,
            scalar_separator[0],
            scalar_narep[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def join_lists_with_scalar(
        Column source_strings,
        object py_separator,
        object py_narep):
    """
    Returns a Column by concatenating Lists of strings row-wise
    in `source_strings` with the specified `py_separator`
    between each string in lists and `<NA>`/`None` values
    are replaced by `py_narep`
    """

    cdef DeviceScalar separator = py_separator.device_value
    cdef DeviceScalar narep = py_narep.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_separator = \
        <const string_scalar*>(separator.get_raw_ptr())
    cdef const string_scalar* scalar_narep = <const string_scalar*>(
        narep.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_join_list_elements(
            source_view,
            scalar_separator[0],
            scalar_narep[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def join_lists_with_column(
        Column source_strings,
        Column separator_strings,
        object py_source_narep,
        object py_separator_narep):
    """
    Returns a Column by concatenating Lists of strings row-wise in
    `source_strings` with a corresponding separator at the same
    position in `separator_strings` and `<NA>`/`None` values in
    `source_strings` are replaced by `py_source_narep` and
    `<NA>`/`None` values in `separator_strings` are replaced
    by `py_separator_narep`
    """

    cdef DeviceScalar source_narep = py_source_narep.device_value
    cdef DeviceScalar separator_narep = py_separator_narep.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef column_view separator_view = separator_strings.view()

    cdef const string_scalar* scalar_source_narep = \
        <const string_scalar*>(source_narep.get_raw_ptr())
    cdef const string_scalar* scalar_separator_narep = <const string_scalar*>(
        separator_narep.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_join_list_elements(
            source_view,
            separator_view,
            scalar_separator_narep[0],
            scalar_source_narep[0]
        ))

    return Column.from_unique_ptr(move(c_result))
