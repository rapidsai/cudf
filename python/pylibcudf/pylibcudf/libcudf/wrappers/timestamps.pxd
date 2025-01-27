# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t


cdef extern from "cudf/wrappers/timestamps.hpp" namespace "cudf" nogil:
    ctypedef int32_t timestamp_D
    ctypedef int64_t timestamp_s
    ctypedef int64_t timestamp_ms
    ctypedef int64_t timestamp_us
    ctypedef int64_t timestamp_ns
