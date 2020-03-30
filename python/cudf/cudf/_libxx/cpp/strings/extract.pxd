# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table cimport table
from libcpp.string cimport string


cdef extern from "cudf/strings/extract.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[table] extract(
        column_view source_strings,
        string pattern) except +
