# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t

from pylibcudf.exception_handler cimport libcudf_exception_handler

cdef extern from "<ratio>" namespace "std" nogil:
    cdef cppclass milli:
        pass

    cdef cppclass micro:
        pass

    cdef cppclass nano:
        pass

    cdef cppclass days:
        pass


cdef extern from "<chrono>" namespace "cuda::std::chrono" nogil:
    cdef cppclass duration[Rep, Period=*]:
        duration() except +libcudf_exception_handler
        duration(int64_t) except +libcudf_exception_handler
        duration(int32_t) except +libcudf_exception_handler


cdef extern from "cudf/wrappers/durations.hpp" namespace "cudf" nogil:
    ctypedef duration[int32_t, days] duration_D
    ctypedef duration[int64_t] duration_s
    ctypedef duration[int64_t, milli] duration_ms
    ctypedef duration[int64_t, micro] duration_us
    ctypedef duration[int64_t, nano] duration_ns
