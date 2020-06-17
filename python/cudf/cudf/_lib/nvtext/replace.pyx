# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.move cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.replace cimport (
    replace_tokens as cpp_replace_tokens,
)
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport Scalar


def replace_tokens(Column strings,
                   Column targets,
                   Column replacements,
                   Scalar delimiter):
    """
    The `targets` tokens are searched for within each `strings`
    in the Column and replaced with the corresponding `replacements`
    if found. Tokens are identified by the `delimiter` character
    provided.
    """

    cdef column_view c_strings = strings.view()
    cdef column_view c_targets = targets.view()
    cdef column_view c_replacements = replacements.view()

    cdef string_scalar* c_delimiter = <string_scalar*>delimiter.c_value.get()
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
