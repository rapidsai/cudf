# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.types cimport size_type

from pylibcudf.nvtext.tokenize import TokenizeVocabulary  # no-cython-lint

from cudf._lib.column cimport Column

from pylibcudf import nvtext


@acquire_spill_lock()
def _tokenize_scalar(Column strings, object py_delimiter):
    return Column.from_pylibcudf(
        nvtext.tokenize.tokenize_scalar(
            strings.to_pylibcudf(mode="read"),
            py_delimiter.device_value.c_value
        )
    )


@acquire_spill_lock()
def _tokenize_column(Column strings, Column delimiters):
    return Column.from_pylibcudf(
        nvtext.tokenize.tokenize_column(
            strings.to_pylibcudf(mode="read"),
            delimiters.to_pylibcudf(mode="read"),
        )
    )


@acquire_spill_lock()
def _count_tokens_scalar(Column strings, object py_delimiter):
    return Column.from_pylibcudf(
        nvtext.tokenize.count_tokens_scalar(
            strings.to_pylibcudf(mode="read"),
            py_delimiter.device_value.c_value
        )
    )


@acquire_spill_lock()
def _count_tokens_column(Column strings, Column delimiters):
    return Column.from_pylibcudf(
        nvtext.tokenize.count_tokens_column(
            strings.to_pylibcudf(mode="read"),
            delimiters.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def character_tokenize(Column strings):
    return Column.from_pylibcudf(
        nvtext.tokenize.character_tokenize(
            strings.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def detokenize(Column strings, Column indices, object py_separator):
    return Column.from_pylibcudf(
        nvtext.tokenize.detokenize(
            strings.to_pylibcudf(mode="read"),
            indices.to_pylibcudf(mode="read"),
            py_separator.device_value.c_value
        )
    )


@acquire_spill_lock()
def tokenize_with_vocabulary(Column strings,
                             object vocabulary,
                             object py_delimiter,
                             size_type default_id):
    return Column.from_pylibcudf(
        nvtext.tokenize.tokenize_with_vocabulary(
            strings.to_pylibcudf(mode="read"),
            vocabulary,
            py_delimiter.device_value.c_value,
            default_id
        )
    )
