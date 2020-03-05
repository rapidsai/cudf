# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.scalar.scalar cimport string_scalar
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table cimport table

cdef extern from "cudf/strings/split/partition.hpp" namespace \
        "cudf::strings" nogil:

    cdef unique_ptr[table] partition(
        column_view source_strings,
        string_scalar delimiter) except +

    cdef unique_ptr[table] rpartition(
        column_view source_strings,
        string_scalar delimiter) except +
