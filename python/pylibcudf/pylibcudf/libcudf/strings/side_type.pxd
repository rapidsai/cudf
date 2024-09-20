# Copyright (c) 2022-2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t


cdef extern from "cudf/strings/side_type.hpp" namespace "cudf::strings" nogil:

    cpdef enum class side_type(int32_t):
        LEFT 'cudf::strings::side_type::LEFT'
        RIGHT 'cudf::strings::side_type::RIGHT'
        BOTH 'cudf::strings::side_type::BOTH'

ctypedef int32_t underlying_type_t_side_type
