# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.strings.regex_program cimport regex_program


cdef extern from "cudf/strings/findall.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] findall(
        column_view source_strings,
        regex_program) except +
