# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.column.column cimport column

cdef extern from "cudf/strings/substring.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] slice_strings(
        column_view source_strings,
        size_type start,
        size_type end,
        size_type step) except +

    cdef unique_ptr[column] slice_strings(
        column_view source_strings,
        column_view starts,
        column_view stops) except +
