# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.replace cimport (
    replace_tokens as cpp_replace_tokens,
    filter_tokens as cpp_filter_tokens,
)
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar


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
