# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.move cimport move

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.nvtext.normalize cimport (
    normalize_spaces as cpp_normalize_spaces
)
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar


def normalize_spaces(Column strings, int ngrams):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_normalize_spaces(c_strings))

    return Column.from_unique_ptr(move(c_result))
