# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t


cdef extern from "cudf/wrappers/durations.hpp" namespace "cudf" nogil:
    ctypedef int32_t duration_D
    ctypedef int64_t duration_s
    ctypedef int64_t duration_ms
    ctypedef int64_t duration_us
    ctypedef int64_t duration_ns
