# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from enum import IntEnum

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport Column

from cudf._lib.cpp.nvtext.stemmer cimport (
    porter_stemmer_measure as cpp_porter_stemmer_measure,
    is_letter as cpp_is_letter,
    letter_type as letter_type
)
from cudf._lib.cpp.nvtext.stemmer cimport underlying_type_t_letter_type


class LetterType(IntEnum):
    CONSONANT = <underlying_type_t_letter_type> letter_type.CONSONANT
    VOWEL = <underlying_type_t_letter_type> letter_type.VOWEL


def porter_stemmer_measure(Column strings):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_porter_stemmer_measure(c_strings))

    return Column.from_unique_ptr(move(c_result))


def is_letter(Column strings,
              object ltype,
              size_type index):
    cdef column_view c_strings = strings.view()
    cdef letter_type c_ltype = <letter_type>(
        <underlying_type_t_letter_type> ltype
    )
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_is_letter(c_strings, c_ltype, index))

    return Column.from_unique_ptr(move(c_result))


def is_letter_multi(Column strings,
                    object ltype,
                    Column indices):
    cdef column_view c_strings = strings.view()
    cdef column_view c_indices = indices.view()
    cdef letter_type c_ltype = <letter_type>(
        <underlying_type_t_letter_type> ltype
    )
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_is_letter(c_strings, c_ltype, c_indices))

    return Column.from_unique_ptr(move(c_result))
