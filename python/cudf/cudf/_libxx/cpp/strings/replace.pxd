# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.types cimport size_type
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.scalar.scalar cimport string_scalar
from cudf._libxx.cpp.table.table cimport table
from libcpp.string cimport string
from libc.stdint cimport int32_t


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

    cdef unique_ptr[column] replace(
        column_view source_strings,
        string_scalar target,
        string_scalar repl,
        int32_t maxrepl) except +

    cdef unique_ptr[column] replace(
        column_view source_strings,
        column_view target_strings,
        column_view repl_strings) except +
