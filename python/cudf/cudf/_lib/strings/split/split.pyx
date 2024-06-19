# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.regex_flags cimport regex_flags
from cudf._lib.pylibcudf.libcudf.strings.regex_program cimport regex_program
from cudf._lib.pylibcudf.libcudf.strings.split.split cimport (
    rsplit as cpp_rsplit,
    rsplit_re as cpp_rsplit_re,
    rsplit_record as cpp_rsplit_record,
    rsplit_record_re as cpp_rsplit_record_re,
    split as cpp_split,
    split_re as cpp_split_re,
    split_record as cpp_split_record,
    split_record_re as cpp_split_record_re,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport data_from_unique_ptr


@acquire_spill_lock()
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


@acquire_spill_lock()
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


@acquire_spill_lock()
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


@acquire_spill_lock()
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


@acquire_spill_lock()
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
    cdef regex_flags c_flags = regex_flags.DEFAULT
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(pattern_string, c_flags))
        c_result = move(cpp_split_re(
            source_view,
            dereference(c_prog),
            maxsplit
        ))

    return data_from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )


@acquire_spill_lock()
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
    cdef regex_flags c_flags = regex_flags.DEFAULT
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(pattern_string, c_flags))
        c_result = move(cpp_rsplit_re(
            source_view,
            dereference(c_prog),
            maxsplit
        ))

    return data_from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )


@acquire_spill_lock()
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
    cdef regex_flags c_flags = regex_flags.DEFAULT
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(pattern_string, c_flags))
        c_result = move(cpp_split_record_re(
            source_view,
            dereference(c_prog),
            maxsplit
        ))

    return Column.from_unique_ptr(
        move(c_result),
    )


@acquire_spill_lock()
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
    cdef regex_flags c_flags = regex_flags.DEFAULT
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(pattern_string, c_flags))
        c_result = move(cpp_rsplit_record_re(
            source_view,
            dereference(c_prog),
            maxsplit
        ))

    return Column.from_unique_ptr(
        move(c_result),
    )
