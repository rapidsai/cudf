# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t

from cudf._lib.pylibcudf.libcudf.types cimport int128


cdef extern from "cudf/fixed_point/fixed_point.hpp" namespace "numeric" nogil:
    # cython type stub to help resolve to numeric::decimal64
    ctypedef int64_t decimal64
    # cython type stub to help resolve to numeric::decimal32
    ctypedef int64_t decimal32
    # cython type stub to help resolve to numeric::decimal128
    ctypedef int128 decimal128

    cdef cppclass scale_type:
        scale_type(int32_t)
