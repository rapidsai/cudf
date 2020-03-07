# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.types cimport size_type
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.scalar.scalar cimport string_scalar


cdef extern from "cudf/strings/contains.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] contains_re(
        column_view source_strings,
        string pattern) except +

    cdef unique_ptr[column] count_re(
        column_view source_strings,
        string pattern) except +
