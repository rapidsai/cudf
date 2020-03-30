# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.cpp.scalar.scalar cimport string_scalar
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table cimport table

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
