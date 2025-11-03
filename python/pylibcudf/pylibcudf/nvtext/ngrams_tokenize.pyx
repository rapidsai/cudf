# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.nvtext.ngrams_tokenize cimport (
    ngrams_tokenize as cpp_ngrams_tokenize,
)
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

__all__ = ["ngrams_tokenize"]

cpdef Column ngrams_tokenize(
    Column input,
    size_type ngrams,
    Scalar delimiter,
    Scalar separator,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
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
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings columns of tokens
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_ngrams_tokenize(
            input.view(),
            ngrams,
            dereference(<const string_scalar*>delimiter.get()),
            dereference(<const string_scalar*>separator.get()),
            stream.view(),
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), stream, mr)
