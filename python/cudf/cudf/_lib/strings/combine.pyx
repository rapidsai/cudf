# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.combine cimport (
    concatenate as cpp_concatenate,
    join_list_elements as cpp_join_list_elements,
    join_strings as cpp_join_strings,
    output_if_empty_list,
    separator_on_nulls,
)
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport table_view_from_columns


@acquire_spill_lock()
def concatenate(list source_strings,
                object sep,
                object na_rep):
    """
    Returns a Column by concatenating strings column-wise in `source_strings`
    with the specified `sep` between each column and
    `na`/`None` values are replaced by `na_rep`
    """
    cdef DeviceScalar separator = sep.device_value
    cdef DeviceScalar narep = na_rep.device_value

    cdef unique_ptr[column] c_result
    cdef table_view source_view = table_view_from_columns(source_strings)

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


@acquire_spill_lock()
def join(Column source_strings,
         object sep,
         object na_rep):
    """
    Returns a Column by concatenating strings row-wise in `source_strings`
    with the specified `sep` between each column and
    `na`/`None` values are replaced by `na_rep`
    """

    cdef DeviceScalar separator = sep.device_value
    cdef DeviceScalar narep = na_rep.device_value

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


@acquire_spill_lock()
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
            scalar_narep[0],
            separator_on_nulls.YES,
            output_if_empty_list.NULL_ELEMENT
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
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
            scalar_source_narep[0],
            separator_on_nulls.YES,
            output_if_empty_list.NULL_ELEMENT
        ))

    return Column.from_unique_ptr(move(c_result))
