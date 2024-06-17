# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.nvtext.replace cimport (
    filter_tokens as cpp_filter_tokens,
    replace_tokens as cpp_replace_tokens,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar


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

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef column_view c_strings = strings.view()
    cdef column_view c_targets = targets.view()
    cdef column_view c_replacements = replacements.view()

    cdef const string_scalar* c_delimiter = <const string_scalar*>delimiter\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_replace_tokens(
                c_strings,
                c_targets,
                c_replacements,
                c_delimiter[0],
            )
        )

    return Column.from_unique_ptr(move(c_result))


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

    cdef DeviceScalar replacement = py_replacement.device_value
    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef column_view c_strings = strings.view()
    cdef const string_scalar* c_repl = <const string_scalar*>replacement\
        .get_raw_ptr()
    cdef const string_scalar* c_delimiter = <const string_scalar*>delimiter\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_filter_tokens(
                c_strings,
                min_token_length,
                c_repl[0],
                c_delimiter[0],
            )
        )

    return Column.from_unique_ptr(move(c_result))
