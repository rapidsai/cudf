# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t, int64_t
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport int128 as int128_t


cdef extern from "cudf/fixed_point/fixed_point.hpp" namespace "numeric" nogil:
    cdef cppclass scale_type:
        scale_type(int32_t)

    cdef cppclass decimal32:
        decimal32(int32_t& value, scale_type& scale)

    cdef cppclass decimal64:
        decimal64(int64_t& value, scale_type& scale)

    cdef cppclass decimal128:
        decimal128(int128_t& value, scale_type& scale)
        int128_t value()
        scale_type scale()
