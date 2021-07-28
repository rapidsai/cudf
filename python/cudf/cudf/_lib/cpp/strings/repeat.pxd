# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type


cdef extern from "cudf/strings/repeat_strings.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] repeat_strings(
        column_view source_strings,
        size_type repeat) except +
