# Copyright (c) 2019-2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.strings.findall cimport findall as cpp_findall
from cudf._lib.pylibcudf.libcudf.strings.regex_flags cimport regex_flags
from cudf._lib.pylibcudf.libcudf.strings.regex_program cimport regex_program


@acquire_spill_lock()
def findall(Column source_strings, object pattern, uint32_t flags):
    """
    Returns data with all non-overlapping matches of `pattern`
    in each string of `source_strings` as a lists column.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string pattern_string = <string>str(pattern).encode()
    cdef regex_flags c_flags = <regex_flags>flags
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(pattern_string, c_flags))
        c_result = move(cpp_findall(
            source_view,
            dereference(c_prog)
        ))

    return Column.from_unique_ptr(move(c_result))
