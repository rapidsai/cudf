# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, uint32_t
from libcpp cimport bool


cdef extern from "cudf/types.hpp" namespace "cudf" nogil:
    ctypedef int32_t size_type
    ctypedef uint32_t bitmask_type
    ctypedef uint32_t char_utf8

    # A Hack to let cython compile with __int128_t symbol
    # https://stackoverflow.com/a/27609033
    ctypedef int int128 "__int128_t"

    cpdef enum class mask_state(int32_t):
        UNALLOCATED
        UNINITIALIZED
        ALL_VALID
        ALL_NULL

    cpdef enum class order(bool):
        ASCENDING
        DESCENDING

    cpdef enum class null_order(bool):
        AFTER
        BEFORE

    cpdef enum class sorted(bool):
        NO
        YES

    cdef cppclass order_info:
        sorted is_sorted
        order ordering
        null_order null_ordering

    cpdef enum class null_policy(bool):
        EXCLUDE
        INCLUDE

    cpdef enum class nan_policy(bool):
        NAN_IS_NULL
        NAN_IS_VALID

    cpdef enum class null_equality(bool):
        EQUAL
        UNEQUAL

    cpdef enum class nan_equality(bool):
        ALL_EQUAL
        UNEQUAL

    cpdef enum class type_id(int32_t):
        EMPTY
        INT8
        INT16
        INT32
        INT64
        UINT8
        UINT16
        UINT32
        UINT64
        FLOAT32
        FLOAT64
        BOOL8
        TIMESTAMP_DAYS
        TIMESTAMP_SECONDS
        TIMESTAMP_MILLISECONDS
        TIMESTAMP_MICROSECONDS
        TIMESTAMP_NANOSECONDS
        DICTIONARY32
        STRING
        LIST
        STRUCT
        NUM_TYPE_IDS
        DURATION_SECONDS
        DURATION_MILLISECONDS
        DURATION_MICROSECONDS
        DURATION_NANOSECONDS
        DECIMAL32
        DECIMAL64
        DECIMAL128

    cdef cppclass data_type:
        data_type() except +
        data_type(const data_type&) except +
        data_type(type_id id) except +
        data_type(type_id id, int32_t scale) except +
        type_id id() noexcept
        int32_t scale() noexcept
        bool operator==(const data_type&, const data_type&) noexcept

    cpdef enum class interpolation(int32_t):
        LINEAR
        LOWER
        HIGHER
        MIDPOINT
        NEAREST
