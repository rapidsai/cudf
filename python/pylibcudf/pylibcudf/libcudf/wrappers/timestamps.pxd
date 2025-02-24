# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libc.time cimport time_t

from libc.stdint cimport int32_t, int64_t


cdef extern from "<chrono>" namespace "cuda::std::chrono" nogil:
    cdef cppclass duration:
        pass
    cdef cppclass nanoseconds:
        pass
    cdef cppclass microseconds:
        pass
    cdef cppclass time_point[T, Duration=*]:
        pass

    cdef cppclass system_clock:
        @staticmethod
        time_point[system_clock, nanoseconds] from_time_t(time_t t) except +

    cdef time_point[T, U_TO] time_point_cast[T, U_TO, U_FROM](
        time_point[T, U_FROM] tp
    ) except +

cdef extern from "cudf/wrappers/timestamps.hpp" namespace "cudf" nogil:
    ctypedef int32_t timestamp_D
    ctypedef int64_t timestamp_s
    ctypedef int64_t timestamp_ms
    ctypedef time_point[system_clock, microseconds] timestamp_us
    ctypedef int64_t timestamp_ns
