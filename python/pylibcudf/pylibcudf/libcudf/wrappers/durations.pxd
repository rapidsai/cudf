# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int64_t


cdef extern from "cudf/wrappers/durations.hpp" namespace "cudf" nogil:
    ctypedef int64_t duration_s
    ctypedef int64_t duration_ms
    ctypedef int64_t duration_us
    ctypedef int64_t duration_ns
