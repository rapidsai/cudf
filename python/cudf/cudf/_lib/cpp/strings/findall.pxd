# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table


cdef extern from "cudf/strings/findall.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[table] findall(
        const column_view& source_strings,
        const string& pattern) except +

    cdef unique_ptr[column] findall_record(
        const column_view& source_strings,
        const string& pattern) except +
