# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libc.time cimport time_t

from libc.stdint cimport int32_t, int64_t
from pylibcudf.libcudf.wrappers.durations cimport (
    duration_s,
    duration_ms,
    duration_us,
    duration_ns,
)

cdef extern from "<chrono>" namespace "cuda::std::chrono" nogil:
    cdef cppclass nanoseconds:
        pass
    cdef cppclass time_point[T, Duration=*]:
        pass

    cdef cppclass system_clock:
        @staticmethod
        time_point[system_clock, nanoseconds] from_time_t(time_t t) except +

    cdef time_point[C, TO_DUR] time_point_cast[TO_DUR, C, DUR](
        time_point[C, DUR] tp
    ) except +

cdef extern from "cudf/wrappers/timestamps.hpp" namespace "cudf" nogil:
    ctypedef int32_t timestamp_D
    ctypedef int64_t timestamp_s
    ctypedef int64_t timestamp_ms
    ctypedef time_point[system_clock, duration_us] timestamp_us
    ctypedef int64_t timestamp_ns
