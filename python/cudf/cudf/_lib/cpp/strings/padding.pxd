# Copyright (c) 2020, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type


cdef extern from "cudf/strings/padding.hpp" namespace "cudf::strings" nogil:
    ctypedef enum pad_side:
        LEFT 'cudf::strings::pad_side::LEFT'
        RIGHT 'cudf::strings::pad_side::RIGHT'
        BOTH 'cudf::strings::pad_side::BOTH'

    cdef unique_ptr[column] pad(
        column_view source_strings,
        size_type width,
        pad_side side,
        string fill_char) except +

    cdef unique_ptr[column] zfill(
        column_view source_strings,
        size_type width) except +

ctypedef int32_t underlying_type_t_pad_side
