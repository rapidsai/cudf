# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.types cimport size_type


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
