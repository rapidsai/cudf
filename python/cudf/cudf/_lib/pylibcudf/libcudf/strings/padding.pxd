# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.side_type cimport side_type
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/padding.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] pad(
        column_view source_strings,
        size_type width,
        side_type side,
        string fill_char) except +

    cdef unique_ptr[column] zfill(
        column_view source_strings,
        size_type width) except +
