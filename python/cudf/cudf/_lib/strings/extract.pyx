# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.strings.extract cimport extract as cpp_extract
from cudf._lib.pylibcudf.libcudf.strings.regex_flags cimport regex_flags
from cudf._lib.pylibcudf.libcudf.strings.regex_program cimport regex_program
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.utils cimport data_from_unique_ptr


@acquire_spill_lock()
def extract(Column source_strings, object pattern, uint32_t flags):
    """
    Returns data which contains extracted capture groups provided in
    `pattern` for all `source_strings`.
    The returning data contains one row for each subject string,
    and one column for each group.
    """
    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()

    cdef string pattern_string = <string>str(pattern).encode()
    cdef regex_flags c_flags = <regex_flags>flags
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(pattern_string, c_flags))
        c_result = move(cpp_extract(
            source_view,
            dereference(c_prog)
        ))

    return data_from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )
