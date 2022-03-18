# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar


cdef extern from "cudf/strings/strip.hpp" namespace "cudf::strings" nogil:
    ctypedef enum strip_type:
        LEFT 'cudf::strings::strip_type::LEFT'
        RIGHT 'cudf::strings::strip_type::RIGHT'
        BOTH 'cudf::strings::strip_type::BOTH'

    cdef unique_ptr[column] strip(
        column_view source_strings,
        strip_type stype,
        string_scalar to_strip) except +
