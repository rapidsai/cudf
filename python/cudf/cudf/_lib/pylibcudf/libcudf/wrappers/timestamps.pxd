# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int64_t


cdef extern from "cudf/wrappers/timestamps.hpp" namespace "cudf" nogil:
    ctypedef int64_t timestamp_s
    ctypedef int64_t timestamp_ms
    ctypedef int64_t timestamp_us
    ctypedef int64_t timestamp_ns
