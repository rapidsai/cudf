# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type


cdef extern from "cudf/strings/split/split.hpp" namespace \
        "cudf::strings" nogil:

    cdef unique_ptr[table] split(
        column_view source_strings,
        string_scalar delimiter,
        size_type maxsplit) except +

    cdef unique_ptr[table] rsplit(
        column_view source_strings,
        string_scalar delimiter,
        size_type maxsplit) except +

    cdef unique_ptr[column] split_record(
        column_view source_strings,
        string_scalar delimiter,
        size_type maxsplit) except +

    cdef unique_ptr[column] rsplit_record(
        column_view source_strings,
        string_scalar delimiter,
        size_type maxsplit) except +
