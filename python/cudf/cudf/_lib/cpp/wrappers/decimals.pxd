# Copyright (c) 2021, NVIDIA CORPORATION.
from libc.stdint cimport int32_t, int64_t


cdef extern from "cudf/fixed_point/fixed_point.hpp" namespace "numeric" nogil:
    # cython type stub to help resolve to numeric::decimal64
    ctypedef int64_t decimal64
    # cython type stub to help resolve to numeric::decimal32
    ctypedef int64_t decimal32

    cdef cppclass scale_type:
        scale_type(int32_t)
