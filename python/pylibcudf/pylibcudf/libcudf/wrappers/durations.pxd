# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t

cdef extern from "std" nogil:
    cdef cppclass milli:
        pass

    cdef cppclass micro:
        pass

    cdef cppclass nano:
        pass

cdef extern from "<chrono>" namespace "cuda::std::chrono" nogil:
    cdef cppclass duration[Rep, Period=*]:
        pass


cdef extern from "cudf/wrappers/durations.hpp" namespace "cudf" nogil:
    ctypedef int32_t duration_D
    ctypedef duration[int64_t] duration_s
    ctypedef duration[int64_t, milli] duration_ms
    ctypedef duration[int64_t, micro] duration_us
    ctypedef duration[int64_t, nano] duration_ns
