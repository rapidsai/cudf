# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.strings.split.split cimport (
    rsplit as cpp_rsplit,
    rsplit_re as cpp_rsplit_re,
    rsplit_record as cpp_rsplit_record,
    rsplit_record_re as cpp_rsplit_record_re,
    split as cpp_split,
    split_re as cpp_split_re,
    split_record as cpp_split_record,
    split_record_re as cpp_split_record_re,
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport data_from_unique_ptr


def split(Column source_strings,
          object py_delimiter,
          size_type maxsplit):
    """
    Returns data by splitting the `source_strings`
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

    return data_from_unique_ptr(
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
    Returns data by splitting the `source_strings`
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

    return data_from_unique_ptr(
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


def split_re(Column source_strings,
             object pattern,
             size_type maxsplit):
    """
    Returns data by splitting the `source_strings`
    column around the delimiters identified by `pattern`.
    """
    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()
    cdef string pattern_string = <string>str(pattern).encode()

    with nogil:
        c_result = move(cpp_split_re(
            source_view,
            pattern_string,
            maxsplit
        ))

    return data_from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )


def rsplit_re(Column source_strings,
              object pattern,
              size_type maxsplit):
    """
    Returns data by splitting the `source_strings`
    column around the delimiters identified by `pattern`.
    The delimiters are searched starting from the end of each string.
    """
    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()
    cdef string pattern_string = <string>str(pattern).encode()

    with nogil:
        c_result = move(cpp_rsplit_re(
            source_view,
            pattern_string,
            maxsplit
        ))

    return data_from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )


def split_record_re(Column source_strings,
                    object pattern,
                    size_type maxsplit):
    """
    Returns a Column by splitting the `source_strings`
    column around the delimiters identified by `pattern`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef string pattern_string = <string>str(pattern).encode()

    with nogil:
        c_result = move(cpp_split_record_re(
            source_view,
            pattern_string,
            maxsplit
        ))

    return Column.from_unique_ptr(
        move(c_result),
    )


def rsplit_record_re(Column source_strings,
                     object pattern,
                     size_type maxsplit):
    """
    Returns a Column by splitting the `source_strings`
    column around the delimiters identified by `pattern`.
    The delimiters are searched starting from the end of each string.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef string pattern_string = <string>str(pattern).encode()

    with nogil:
        c_result = move(cpp_rsplit_record_re(
            source_view,
            pattern_string,
            maxsplit
        ))

    return Column.from_unique_ptr(
        move(c_result),
    )
