# Copyright (c) 2021, NVIDIA CORPORATION.
from libc.stdint cimport int64_t, int32_t

cdef extern from "cudf/fixed_point/fixed_point.hpp" namespace "numeric" nogil:
    # cython type stub to help resolve to numeric::decimal64
    ctypedef int64_t decimal64

    cdef cppclass scale_type:
        scale_type(int32_t)
