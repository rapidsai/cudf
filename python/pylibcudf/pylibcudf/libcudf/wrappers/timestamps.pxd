# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t, int64_t
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.wrappers.durations cimport (
    duration_s,
    duration_ms,
    duration_us,
    duration_ns,
    duration_D,
)


cdef extern from "<chrono>" namespace "cuda::std::chrono" nogil:
    cdef cppclass time_point[T, U]:
        time_point() except +libcudf_exception_handler
        time_point(duration_s) except +libcudf_exception_handler
        time_point(duration_ms) except +libcudf_exception_handler
        time_point(duration_ns) except +libcudf_exception_handler
        time_point(duration_us) except +libcudf_exception_handler
        time_point(duration_D) except +libcudf_exception_handler

    cdef cppclass system_clock:
        pass


cdef extern from "cudf/wrappers/timestamps.hpp" namespace "cudf" nogil:
    ctypedef time_point[system_clock, duration_D] timestamp_D
    ctypedef time_point[system_clock, duration_s] timestamp_s
    ctypedef time_point[system_clock, duration_ms] timestamp_ms
    ctypedef time_point[system_clock, duration_us] timestamp_us
    ctypedef time_point[system_clock, duration_ns] timestamp_ns
