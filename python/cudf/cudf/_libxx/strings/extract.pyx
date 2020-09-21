# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.move cimport move
from cudf._libxx.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.table cimport Table

from cudf._libxx.cpp.column.column cimport column

from cudf._libxx.cpp.strings.extract cimport (
    extract as cpp_extract
)
from libcpp.string cimport string


def extract(Column source_strings, object pattern):
    """
    Returns a Table which contains extracted capture groups provided in
    `pattern` for all `source_strings`.
    The returning Table contains one row for each subject string,
    and one column for each group.
    """
    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()

    cdef string pattern_string = <string>str(pattern).encode()

    with nogil:
        c_result = move(cpp_extract(
            source_view,
            pattern_string
        ))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )
