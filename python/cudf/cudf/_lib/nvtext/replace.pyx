# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column
from pylibcudf import nvtext


@acquire_spill_lock()
def replace_tokens(Column strings,
                   Column targets,
                   Column replacements,
                   object py_delimiter):
    """
    The `targets` tokens are searched for within each `strings`
    in the Column and replaced with the corresponding `replacements`
    if found. Tokens are identified by the `py_delimiter` character
    provided.
    """

    return Column.from_pylibcudf(
        nvtext.replace.replace_tokens(
            strings.to_pylibcudf(mode="read"),
            targets.to_pylibcudf(mode="read"),
            replacements.to_pylibcudf(mode="read"),
            py_delimiter.device_value.c_value,
        )
    )


@acquire_spill_lock()
def filter_tokens(Column strings,
                  size_type min_token_length,
                  object py_replacement,
                  object py_delimiter):
    """
    Tokens smaller than `min_token_length` are removed from `strings`
    in the Column and optionally replaced with the corresponding
    `py_replacement` string. Tokens are identified by the `py_delimiter`
    character provided.
    """

    return Column.from_pylibcudf(
        nvtext.replace.filter_tokens(
            strings.to_pylibcudf(mode="read"),
            min_token_length,
            py_replacement.device_value.c_value,
            py_delimiter.device_value.c_value,
        )
    )
