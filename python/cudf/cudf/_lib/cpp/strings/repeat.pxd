# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type


cdef extern from "cudf/strings/repeat_strings.hpp" namespace "cudf::strings" \
        nogil:

    cdef unique_ptr[column] repeat_strings(
        column_view strings,
        size_type repeat) except +

    cdef unique_ptr[column] repeat_strings(
        column_view strings,
        column_view repeats) except +
