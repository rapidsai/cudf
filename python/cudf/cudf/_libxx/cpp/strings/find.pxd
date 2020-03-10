# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.scalar.scalar cimport string_scalar


cdef extern from "cudf/strings/find.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] contains(
        column_view source_strings,
        string_scalar target) except +
