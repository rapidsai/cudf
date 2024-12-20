# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t


cdef extern from "cudf/fixed_point/fixed_point.hpp" namespace "numeric" nogil:
    cdef cppclass scale_type:
        scale_type(int32_t)
