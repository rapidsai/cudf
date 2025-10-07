# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
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
from pylibcudf.utils cimport _get_stream
from rmm.pylibrmm.stream cimport Stream

__all__ = [
    "generate_ngrams",
    "generate_character_ngrams",
    "hash_character_ngrams",
]

cpdef Column generate_ngrams(
    Column input, size_type ngrams, Scalar separator, Stream stream=None
):
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
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings columns of tokens
    """
    cdef column_view c_strings = input.view()
    cdef const string_scalar* c_separator = <const string_scalar*>separator.c_obj.get()
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_generate_ngrams(
            c_strings,
            ngrams,
            c_separator[0],
            stream.view()
        )
    return Column.from_libcudf(move(c_result), stream)


cpdef Column generate_character_ngrams(
    Column input, size_type ngrams = 2, Stream stream=None
):
    """
    Returns a lists column of ngrams of characters within each string.

    For details, see :cpp:func:`generate_character_ngrams`

    Parameters
    ----------
    input : Column
        Input strings
    ngram : size_type
        The ngram number to generate
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Lists column of strings
    """
    cdef column_view c_strings = input.view()
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_generate_character_ngrams(
            c_strings,
            ngrams,
            stream.view()
        )
    return Column.from_libcudf(move(c_result), stream)


cpdef Column hash_character_ngrams(
    Column input, size_type ngrams, uint32_t seed, Stream stream=None
):
    """
    Returns a lists column of hash values of the characters in each string

    For details, see :cpp:func:`hash_character_ngrams`

    Parameters
    ----------
    input : Column
        Input strings
    ngram : size_type
        The ngram number to generate
    seed : uint32_t
        Seed used for the hash algorithm
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Lists column of hash values
    """
    cdef column_view c_strings = input.view()
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_hash_character_ngrams(
            c_strings,
            ngrams,
            seed,
            stream.view()
        )
    return Column.from_libcudf(move(c_result), stream)
