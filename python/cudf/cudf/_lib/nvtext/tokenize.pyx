# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.nvtext.tokenize cimport (
    character_tokenize as cpp_character_tokenize,
    count_tokens as cpp_count_tokens,
    detokenize as cpp_detokenize,
    load_vocabulary as cpp_load_vocabulary,
    tokenize as cpp_tokenize,
    tokenize_vocabulary as cpp_tokenize_vocabulary,
    tokenize_with_vocabulary as cpp_tokenize_with_vocabulary,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def _tokenize_scalar(Column strings, object py_delimiter):

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef column_view c_strings = strings.view()
    cdef const string_scalar* c_delimiter = <const string_scalar*>delimiter\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_tokenize(
                c_strings,
                c_delimiter[0],
            )
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def _tokenize_column(Column strings, Column delimiters):
    cdef column_view c_strings = strings.view()
    cdef column_view c_delimiters = delimiters.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_tokenize(
                c_strings,
                c_delimiters
            )
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def _count_tokens_scalar(Column strings, object py_delimiter):

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef column_view c_strings = strings.view()
    cdef const string_scalar* c_delimiter = <const string_scalar*>delimiter\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_count_tokens(
                c_strings,
                c_delimiter[0]
            )
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def _count_tokens_column(Column strings, Column delimiters):
    cdef column_view c_strings = strings.view()
    cdef column_view c_delimiters = delimiters.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_count_tokens(
                c_strings,
                c_delimiters
            )
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def character_tokenize(Column strings):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_character_tokenize(c_strings)
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def detokenize(Column strings, Column indices, object py_separator):

    cdef DeviceScalar separator = py_separator.device_value

    cdef column_view c_strings = strings.view()
    cdef column_view c_indices = indices.view()
    cdef const string_scalar* c_separator = <const string_scalar*>separator\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_detokenize(c_strings, c_indices, c_separator[0])
        )

    return Column.from_unique_ptr(move(c_result))


cdef class TokenizeVocabulary:
    cdef unique_ptr[cpp_tokenize_vocabulary] c_obj

    def __cinit__(self, Column vocab):
        cdef column_view c_vocab = vocab.view()
        with nogil:
            self.c_obj = move(cpp_load_vocabulary(c_vocab))


@acquire_spill_lock()
def tokenize_with_vocabulary(Column strings,
                             TokenizeVocabulary vocabulary,
                             object py_delimiter,
                             size_type default_id):

    cdef DeviceScalar delimiter = py_delimiter.device_value
    cdef column_view c_strings = strings.view()
    cdef const string_scalar* c_delimiter = <const string_scalar*>delimiter\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_tokenize_with_vocabulary(
                c_strings,
                vocabulary.c_obj.get()[0],
                c_delimiter[0],
                default_id
            )
        )

    return Column.from_unique_ptr(move(c_result))
