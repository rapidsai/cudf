# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.regex_flags cimport regex_flags
from cudf._lib.pylibcudf.libcudf.strings.regex_program cimport regex_program
from cudf._lib.pylibcudf.libcudf.strings.replace_re cimport (
    replace_re as cpp_replace_re,
    replace_with_backrefs as cpp_replace_with_backrefs,
)
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def replace_re(Column source_strings,
               object pattern,
               object py_repl,
               size_type n):
    """
    Returns a Column after replacing occurrences regular
    expressions `pattern` with `py_repl` in `source_strings`.
    `n` indicates the number of resplacements to be made from
    start. (-1 indicates all)
    """

    cdef DeviceScalar repl = py_repl.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string pattern_string = <string>str(pattern).encode()
    cdef const string_scalar* scalar_repl = \
        <const string_scalar*>(repl.get_raw_ptr())
    cdef regex_flags c_flags = regex_flags.DEFAULT
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(pattern_string, c_flags))
        c_result = move(cpp_replace_re(
            source_view,
            dereference(c_prog),
            scalar_repl[0],
            n
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
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
    cdef regex_flags c_flags = regex_flags.DEFAULT
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(pattern_string, c_flags))
        c_result = move(cpp_replace_with_backrefs(
            source_view,
            dereference(c_prog),
            repl_string
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
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
