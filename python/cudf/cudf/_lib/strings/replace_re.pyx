# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport Scalar
from cudf._lib.cpp.types cimport size_type
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport string_scalar

from cudf._lib.cpp.strings.replace_re cimport (
    replace_re as cpp_replace_re,
    replace_with_backrefs as cpp_replace_with_backrefs
)
from libcpp.string cimport string


def replace_re(Column source_strings,
               object pattern,
               Scalar repl,
               size_type n):
    """
    Returns a Column after replacing occurrences regular
    expressions `pattern` with `repl` in `source_strings`.
    `n` indicates the number of resplacements to be made from
    start. (-1 indicates all)
    """

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string pattern_string = <string>str(pattern).encode()
    cdef string_scalar* scalar_repl = \
        <string_scalar*>(repl.c_value.get())

    with nogil:
        c_result = move(cpp_replace_re(
            source_view,
            pattern_string,
            scalar_repl[0],
            n
        ))

    return Column.from_unique_ptr(move(c_result))


def replace_with_backrefs(
        Column source_strings,
        object pattern,
        object repl):
    """
    Returns a Column after using the `repl` back-ref template to create
    new string with the extracted elements found using
    `pattern` regular expression in `source_strings`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string pattern_string = <string>str(pattern).encode()
    cdef string repl_string = <string>str(repl).encode()

    with nogil:
        c_result = move(cpp_replace_with_backrefs(
            source_view,
            pattern_string,
            repl_string
        ))

    return Column.from_unique_ptr(move(c_result))


def replace_multi_re(Column source_strings,
                     object patterns,
                     Column repl_strings):
    """
    Returns a Column after replacing occurrences of multiple
    regular expressions `patterns` with their corresponding
    strings in `repl_strings` in `source_strings`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef column_view repl_view = repl_strings.view()

    cdef int pattern_size = len(patterns)
    cdef vector[string] patterns_vector
    patterns_vector.reserve(pattern_size)

    for pattern in patterns:
        patterns_vector.push_back(str.encode(pattern))

    with nogil:
        c_result = move(cpp_replace_re(
            source_view,
            patterns_vector,
            repl_view
        ))

    return Column.from_unique_ptr(move(c_result))
