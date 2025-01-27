# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.generate_ngrams cimport (
    generate_character_ngrams as cpp_generate_character_ngrams,
    generate_ngrams as cpp_generate_ngrams,
    hash_character_ngrams as cpp_hash_character_ngrams,
)
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar

__all__ = [
    "generate_ngrams",
    "generate_character_ngrams",
    "hash_character_ngrams",
]

cpdef Column generate_ngrams(Column input, size_type ngrams, Scalar separator):
    """
    Returns a single column of strings by generating ngrams from a strings column.

    For details, see :cpp:func:`generate_ngrams`

    Parameters
    ----------
    input : Column
        Input strings
    ngram : size_type
        The ngram number to generate
    separator : Scalar
        The string to use for separating ngram tokens

    Returns
    -------
    Column
        New strings columns of tokens
    """
    cdef column_view c_strings = input.view()
    cdef const string_scalar* c_separator = <const string_scalar*>separator.c_obj.get()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_generate_ngrams(
            c_strings,
            ngrams,
            c_separator[0]
        )
    return Column.from_libcudf(move(c_result))


cpdef Column generate_character_ngrams(Column input, size_type ngrams = 2):
    """
    Returns a lists column of ngrams of characters within each string.

    For details, see :cpp:func:`generate_character_ngrams`

    Parameters
    ----------
    input : Column
        Input strings
    ngram : size_type
        The ngram number to generate

    Returns
    -------
    Column
        Lists column of strings
    """
    cdef column_view c_strings = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_generate_character_ngrams(
            c_strings,
            ngrams,
        )
    return Column.from_libcudf(move(c_result))

cpdef Column hash_character_ngrams(Column input, size_type ngrams = 2):
    """
    Returns a lists column of hash values of the characters in each string

    For details, see :cpp:func:`hash_character_ngrams`

    Parameters
    ----------
    input : Column
        Input strings
    ngram : size_type
        The ngram number to generate

    Returns
    -------
    Column
        Lists column of hash values
    """
    cdef column_view c_strings = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_hash_character_ngrams(
            c_strings,
            ngrams,
        )
    return Column.from_libcudf(move(c_result))
