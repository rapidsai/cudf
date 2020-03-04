# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.cpp.scalar.scalar cimport string_scalar
from libcpp.memory cimport unique_ptr
from cudf._libxx.cpp.column.column cimport column

cdef extern from "cudf/strings/replace.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] replace_slice(
        column_view source_strings,
        string_scalar repl,
        size_type start,
        size_type stop) except +
