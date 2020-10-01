# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.generate_ngrams cimport (
    generate_ngrams as cpp_generate_ngrams,
    generate_character_ngrams as cpp_generate_character_ngrams
)
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport Scalar


def generate_ngrams(Column strings, int ngrams, Scalar separator):
    cdef column_view c_strings = strings.view()
    cdef size_type c_ngrams = ngrams
    cdef string_scalar* c_separator = <string_scalar*>separator.c_value.get()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_generate_ngrams(
                c_strings,
                c_ngrams,
                c_separator[0]
            )
        )

    return Column.from_unique_ptr(move(c_result))


def generate_character_ngrams(Column strings, int ngrams):
    cdef column_view c_strings = strings.view()
    cdef size_type c_ngrams = ngrams
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_generate_character_ngrams(
                c_strings,
                c_ngrams
            )
        )

    return Column.from_unique_ptr(move(c_result))
