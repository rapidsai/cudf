# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.ngrams_tokenize cimport ngrams_tokenize as cpp_ngrams_tokenize
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar


cpdef Column ngrams_tokenize(Column input, size_type ngrams, Scalar delimiter, Scalar separator):
    """
    Returns a single column of strings by tokenizing the input strings column
    and then producing ngrams of each string.

    For details, see :cpp:func:`ngrams_tokenize`

    Parameters
    ----------
    input : Column
        Input strings
    ngrams : size_type
        The ngram number to generate
    delimiter : Scalar
        UTF-8 characters used to separate each string into tokens.
        An empty string will separate tokens using whitespace.
    separator : Scalar
        The string to use for separating ngram tokens

    Returns
    -------
    Column
        New strings columns of tokens
    """
    cdef column_view c_strings = input.view()
    cdef const string_scalar* c_delimiter = <const string_scalar*>delimiter.c_obj.get()
    cdef const string_scalar* c_separator = <const string_scalar*>separator.c_obj.get()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_ngrams_tokenize(
                input.view(),
                ngrams,
                c_delimiter[0],
                c_separator[0],
            )
        )
    return Column.from_libcudf(move(c_result))
