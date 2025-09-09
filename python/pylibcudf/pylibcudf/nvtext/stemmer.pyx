# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.nvtext.stemmer cimport (
    is_letter as cpp_is_letter,
    letter_type,
    porter_stemmer_measure as cpp_porter_stemmer_measure,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.utils cimport _get_stream

from pylibcudf.libcudf.nvtext.stemmer import letter_type as LetterType # no-cython-lint
from rmm.pylibrmm.stream cimport Stream

__all__ = ["is_letter", "porter_stemmer_measure", "LetterType"]

cpdef Column is_letter(
    Column input,
    bool check_vowels,
    ColumnOrSize indices,
    Stream stream=None
):
    """
    Returns boolean column indicating if the character
    or characters at the provided character index or
    indices (respectively) are consonants or vowels

    For details, see :cpp:func:`is_letter`

    Parameters
    ----------
    input : Column
        Input strings
    check_vowels : bool
        If true, the check is for vowels. Otherwise the check is
        for consonants.
    indices : Union[Column, size_type]
        The character position(s) to check in each string
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New boolean column.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_is_letter(
            input.view(),
            letter_type.VOWEL if check_vowels else letter_type.CONSONANT,
            indices if ColumnOrSize is size_type else indices.view(),
            stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)


cpdef Column porter_stemmer_measure(Column input, Stream stream=None):
    """
    Returns the Porter Stemmer measurements of a strings column.

    For details, see :cpp:func:`porter_stemmer_measure`

    Parameters
    ----------
    input : Column
        Strings column of words to measure
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of measure values
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_porter_stemmer_measure(input.view(), stream.view())

    return Column.from_libcudf(move(c_result), stream)

LetterType.__str__ = LetterType.__repr__
