# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


cdef extern from "cudf/strings/contains.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] contains_re(
        column_view source_strings,
        string pattern) except +

    cdef unique_ptr[column] count_re(
        column_view source_strings,
        string pattern) except +

    cdef unique_ptr[column] matches_re(
        column_view source_strings,
        string pattern) except +
