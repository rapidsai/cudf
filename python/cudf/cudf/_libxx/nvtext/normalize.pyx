# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.move cimport move

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.nvtext.normalize cimport (
    normalize as cpp_normalize
)
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar


def normalize_spaces(Column strings, int ngrams):
    cdef column_view source_view = strings.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_normalize(c_strings))

    return Column.from_unique_ptr(move(c_result))
