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
from cudf._lib.scalar cimport Scalar


def tokenize(Column strings, object delimiter):
    if isinstance(delimiter, Scalar):
        return _tokenize_scalar(strings, delimiter)

    if isinstance(delimiter, Column):
        return _tokenize_column(strings, delimiter)

    raise TypeError(
        "Expected a Scalar or Column for delimiters, but got {}".format(
            type(delimiter)
        )
    )


def _tokenize_scalar(Column strings, Scalar delimiter):

    cdef column_view c_strings = strings.view()
    cdef string_scalar* c_delimiter = <string_scalar*>delimiter.c_value.get()
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


def count_tokens(Column strings, object delimiter):
    if isinstance(delimiter, Scalar):
        return _count_tokens_scalar(strings, delimiter)

    if isinstance(delimiter, Column):
        return _count_tokens_column(strings, delimiter)

    raise TypeError(
        "Expected a Scalar or Column for delimiters, but got {}".format(
            type(delimiter)
        )
    )


def _count_tokens_scalar(Column strings, Scalar delimiter):
    cdef column_view c_strings = strings.view()
    cdef string_scalar* c_delimiter = <string_scalar*>delimiter.c_value.get()
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


def detokenize(Column strings, Column indices, Scalar separator):
    cdef column_view c_strings = strings.view()
    cdef column_view c_indices = indices.view()
    cdef string_scalar* c_separator = <string_scalar*>separator.c_value.get()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_detokenize(c_strings, c_indices, c_separator[0])
        )

    return Column.from_unique_ptr(move(c_result))
