# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.move cimport move

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.scalar.scalar cimport scalar
from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.nvtext.tokenize cimport (
    tokenize as cpp_tokenize,
    count_tokens as cpp_count_tokens
)
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar


def tokenize(Column strings, Scalar delimiter):
    cdef column_view c_strings = strings.view()
    cdef scalar* c_delimiter = delimiter.c_value.get()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_tokenize(
                c_strings,
                c_ngrams
                c_delimiter[0]
                c_separator[0]
            )
        )

    return Column.from_unique_ptr(move(c_result))


def tokenize(Column strings, Column delimiters):
    cdef column_view c_strings = strings.view()
    cdef column_view c_delimiter = delimiter.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_tokenize(
                c_strings,
                c_ngrams
                c_delimiter[0]
                c_separator[0]
            )
        )

    return Column.from_unique_ptr(move(c_result))


def count_tokens(Column strings, Scalar delimiter):
    cdef column_view c_strings = strings.view()
    cdef scalar* c_delimiter = delimiter.c_value.get()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_count_tokens(
                c_strings,
                c_ngrams
                c_delimiter[0]
                c_separator[0]
            )
        )

    return Column.from_unique_ptr(move(c_result))


def count_tokens(Column strings, Column delimiters):
    cdef column_view c_strings = strings.view()
    cdef column_view c_delimiter = delimiter.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_count_tokens(
                c_strings,
                c_ngrams
                c_delimiter[0]
                c_separator[0]
            )
        )

    return Column.from_unique_ptr(move(c_result))
