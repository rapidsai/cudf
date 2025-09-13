# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.tokenize cimport (
    character_tokenize as cpp_character_tokenize,
    count_tokens as cpp_count_tokens,
    detokenize as cpp_detokenize,
    load_vocabulary as cpp_load_vocabulary,
    tokenize as cpp_tokenize,
    tokenize_with_vocabulary as cpp_tokenize_with_vocabulary,
)
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.utils cimport _get_stream
from rmm.pylibrmm.stream cimport Stream

__all__ = [
    "TokenizeVocabulary",
    "character_tokenize",
    "count_tokens_column",
    "count_tokens_scalar",
    "detokenize",
    "tokenize_column",
    "tokenize_scalar",
    "tokenize_with_vocabulary",
]

cdef class TokenizeVocabulary:
    """The Vocabulary object to be used with ``tokenize_with_vocabulary``.

    For details, see :cpp:class:`cudf::nvtext::tokenize_vocabulary`.
    """
    def __cinit__(self, Column vocab, Stream stream=None):
        cdef column_view c_vocab = vocab.view()
        stream = _get_stream(stream)
        with nogil:
            self.c_obj = move(cpp_load_vocabulary(c_vocab, stream.view()))

    __hash__ = None

cpdef Column tokenize_scalar(Column input, Scalar delimiter=None, Stream stream=None):
    """
    Returns a single column of strings by tokenizing the input
    strings column using the provided characters as delimiters.

    For details, see cpp:func:`cudf::nvtext::tokenize`

    Parameters
    ----------
    input : Column
        Strings column to tokenize
    delimiter : Scalar
        String scalar used to separate individual strings into tokens
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings columns of tokens
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    if delimiter is None:
        delimiter = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), stream.view())
        )

    with nogil:
        c_result = cpp_tokenize(
            input.view(),
            dereference(<const string_scalar*>delimiter.c_obj.get()),
            stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)

cpdef Column tokenize_column(Column input, Column delimiters, Stream stream=None):
    """
    Returns a single column of strings by tokenizing the input
    strings column using multiple strings as delimiters.

    For details, see cpp:func:`cudf::nvtext::tokenize`

    Parameters
    ----------
    input : Column
        Strings column to tokenize
    delimiters : Column
        Strings column used to separate individual strings into tokens
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings columns of tokens
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_tokenize(
            input.view(),
            delimiters.view(),
            stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)

cpdef Column count_tokens_scalar(
    Column input, Scalar delimiter=None, Stream stream=None
):
    """
    Returns the number of tokens in each string of a strings column
    using the provided characters as delimiters.

    For details, see cpp:func:`cudf::nvtext::count_tokens`

    Parameters
    ----------
    input : Column
        Strings column to count tokens
    delimiters : Scalar
        String scalar used to separate each string into tokens
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of token counts
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    if delimiter is None:
        delimiter = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), stream.view())
        )

    with nogil:
        c_result = cpp_count_tokens(
            input.view(),
            dereference(<const string_scalar*>delimiter.c_obj.get()),
            stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)

cpdef Column count_tokens_column(Column input, Column delimiters, Stream stream=None):
    """
    Returns the number of tokens in each string of a strings column
    using multiple strings as delimiters.

    For details, see cpp:func:`cudf::nvtext::count_tokens`

    Parameters
    ----------
    input : Column
        Strings column to count tokens
    delimiters : Column
        Strings column used to separate each string into tokens
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of token counts
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_count_tokens(
            input.view(),
            delimiters.view(),
            stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)

cpdef Column character_tokenize(Column input, Stream stream=None):
    """
    Returns a single column of strings by converting
    each character to a string.

    For details, see cpp:func:`cudf::nvtext::character_tokens`

    Parameters
    ----------
    input : Column
        Strings column to tokenize
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings columns of tokens
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    with nogil:
        c_result = cpp_character_tokenize(input.view(), stream.view())

    return Column.from_libcudf(move(c_result), stream)

cpdef Column detokenize(
    Column input,
    Column row_indices,
    Scalar separator=None,
    Stream stream=None
):
    """
    Creates a strings column from a strings column of tokens
    and an associated column of row ids.

    For details, see cpp:func:`cudf::nvtext::detokenize`

    Parameters
    ----------
    input : Column
        Strings column to detokenize
    row_indices : Column
        The relative output row index assigned for each token in the input column
    separator : Scalar
        String to append after concatenating each token to the proper output row
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings columns of tokens
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    if separator is None:
        separator = Scalar.from_libcudf(
            cpp_make_string_scalar(" ".encode(), stream.view())
        )

    with nogil:
        c_result = cpp_detokenize(
            input.view(),
            row_indices.view(),
            dereference(<const string_scalar*>separator.c_obj.get()),
            stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)

cpdef Column tokenize_with_vocabulary(
    Column input,
    TokenizeVocabulary vocabulary,
    Scalar delimiter,
    size_type default_id=-1,
    Stream stream=None
):
    """
    Returns the token ids for the input string by looking
    up each delimited token in the given vocabulary.

    For details, see cpp:func:`cudf::nvtext::tokenize_with_vocabulary`

    Parameters
    ----------
    input : Column
        Strings column to tokenize
    vocabulary : TokenizeVocabulary
        Used to lookup tokens within ``input``
    delimiter : Scalar
        Used to identify tokens within ``input``
    default_id : size_type
        The token id to be used for tokens not found in the vocabulary; Default is -1
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Lists column of token ids
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_tokenize_with_vocabulary(
            input.view(),
            dereference(vocabulary.c_obj.get()),
            dereference(<const string_scalar*>delimiter.c_obj.get()),
            default_id,
            stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)
