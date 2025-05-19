# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext cimport normalize as cpp_normalize

__all__ = [
    "CharacterNormalizer"
    "normalize_characters",
    "normalize_spaces",
    "characters_normalize"
]

cdef class CharacterNormalizer:
    """The normalizer object to be used with ``normalize_characters``.

    For details, see :cpp:class:`cudf::nvtext::character_normalizer`.
    """
    def __cinit__(self, bool do_lower_case, Column tokens):
        cdef column_view c_tokens = tokens.view()
        with nogil:
            self.c_obj = move(
                cpp_normalize.create_character_normalizer(
                    do_lower_case,
                    c_tokens
                )
            )

    __hash__ = None

cpdef Column normalize_spaces(Column input):
    """
    Returns a new strings column by normalizing the whitespace in
    each string in the input column.

    For details, see :cpp:func:`normalize_spaces`

    Parameters
    ----------
    input : Column
        Input strings

    Returns
    -------
    Column
        New strings columns of normalized strings.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_normalize.normalize_spaces(input.view())

    return Column.from_libcudf(move(c_result))


cpdef Column characters_normalize(Column input, bool do_lower_case):
    """
    Normalizes strings characters for tokenizing.

    For details, see :cpp:func:`normalize_characters`

    Parameters
    ----------
    input : Column
        Input strings
    do_lower_case : bool
        If true, upper-case characters are converted to lower-case
        and accents are stripped from those characters. If false,
        accented and upper-case characters are not transformed.

    Returns
    -------
    Column
        Normalized strings column
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_normalize.normalize_characters(
            input.view(),
            do_lower_case
        )

    return Column.from_libcudf(move(c_result))


cpdef Column normalize_characters(Column input, CharacterNormalizer normalizer):
    """
    Normalizes strings characters for tokenizing.

    For details, see :cpp:func:`normalize_characters`

    Parameters
    ----------
    input : Column
        Input strings
    normalizer : CharacterNormalizer
        Normalizer object used for modifying the input column text

    Returns
    -------
    Column
        Normalized strings column
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_normalize.normalize_characters(
            input.view(),
            dereference(normalizer.c_obj.get())
        )

    return Column.from_libcudf(move(c_result))
