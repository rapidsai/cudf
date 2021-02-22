# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.tokenize cimport (
    tokenize as cpp_tokenize,
    detokenize as cpp_detokenize,
    count_tokens as cpp_count_tokens,
    character_tokenize as cpp_character_tokenize
)
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar


def _tokenize_scalar(Column strings, object py_delimiter):

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef column_view c_strings = strings.view()
    cdef const string_scalar* c_delimiter = <const string_scalar*>delimiter\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_tokenize(
                c_strings,
                c_delimiter[0],
            )
        )

    return Column.from_unique_ptr(move(c_result))


def _tokenize_column(Column strings, Column delimiters):
    cdef column_view c_strings = strings.view()
    cdef column_view c_delimiters = delimiters.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_tokenize(
                c_strings,
                c_delimiters
            )
        )

    return Column.from_unique_ptr(move(c_result))


def _count_tokens_scalar(Column strings, object py_delimiter):

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef column_view c_strings = strings.view()
    cdef const string_scalar* c_delimiter = <const string_scalar*>delimiter\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_count_tokens(
                c_strings,
                c_delimiter[0]
            )
        )

    return Column.from_unique_ptr(move(c_result))


def _count_tokens_column(Column strings, Column delimiters):
    cdef column_view c_strings = strings.view()
    cdef column_view c_delimiters = delimiters.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_count_tokens(
                c_strings,
                c_delimiters
            )
        )

    return Column.from_unique_ptr(move(c_result))


def character_tokenize(Column strings):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_character_tokenize(c_strings)
        )

    return Column.from_unique_ptr(move(c_result))


def detokenize(Column strings, Column indices, object py_separator):

    cdef DeviceScalar separator = py_separator.device_value

    cdef column_view c_strings = strings.view()
    cdef column_view c_indices = indices.view()
    cdef const string_scalar* c_separator = <const string_scalar*>separator\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_detokenize(c_strings, c_indices, c_separator[0])
        )

    return Column.from_unique_ptr(move(c_result))
