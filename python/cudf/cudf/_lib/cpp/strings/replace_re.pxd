# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type


cdef extern from "cudf/strings/replace_re.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] replace_re(
        column_view source_strings,
        string pattern,
        string_scalar repl,
        size_type maxrepl) except +

    cdef unique_ptr[column] replace_with_backrefs(
        column_view source_strings,
        string pattern,
        string repl) except +

    cdef unique_ptr[column] replace_re(
        column_view source_strings,
        vector[string] patterns,
        column_view repls) except +
