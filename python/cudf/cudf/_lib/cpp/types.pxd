# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, uint32_t


cdef extern from "cudf/types.hpp" namespace "cudf" nogil:
    # The declaration below is to work around
    # https://github.com/cython/cython/issues/5637
    """
    #define __PYX_ENUM_CLASS_DECL enum
    """
    ctypedef int32_t size_type
    ctypedef uint32_t bitmask_type
    ctypedef uint32_t char_utf8

    ctypedef enum mask_state:
        UNALLOCATED "cudf::mask_state::UNALLOCATED"
        UNINITIALIZED "cudf::mask_state::UNINITIALIZED"
        ALL_VALID "cudf::mask_state::ALL_VALID"
        ALL_NULL "cudf::mask_state::ALL_NULL"

    ctypedef enum order "cudf::order":
        ASCENDING "cudf::order::ASCENDING"
        DESCENDING "cudf::order::DESCENDING"

    ctypedef enum null_order "cudf::null_order":
        AFTER "cudf::null_order::AFTER"
        BEFORE "cudf::null_order::BEFORE"

    ctypedef enum sorted "cudf::sorted":
        NO "cudf::sorted::NO"
        YES "cudf::sorted::YES"

    cdef cppclass order_info:
        sorted is_sorted
        order ordering
        null_order null_ordering

    ctypedef enum null_policy "cudf::null_policy":
        EXCLUDE "cudf::null_policy::EXCLUDE"
        INCLUDE "cudf::null_policy::INCLUDE"

    ctypedef enum nan_policy "cudf::nan_policy":
        NAN_IS_NULL  "cudf::nan_policy::NAN_IS_NULL"
        NAN_IS_VALID "cudf::nan_policy::NAN_IS_VALID"

    ctypedef enum null_equality "cudf::null_equality":
        EQUAL "cudf::null_equality::EQUAL"
        UNEQUAL "cudf::null_equality::UNEQUAL"

    ctypedef enum nan_equality "cudf::nan_equality":
        # These names differ from the C++ names due to Cython warnings if
        # "UNEQUAL" is declared by both null_equality and nan_equality.
        ALL_EQUAL "cudf::nan_equality::ALL_EQUAL"
        NANS_UNEQUAL "cudf::nan_equality::UNEQUAL"

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
        type_id id() except +
        int32_t scale() except +

cdef extern from "cudf/types.hpp" namespace "cudf" nogil:
    ctypedef enum interpolation:
        LINEAR "cudf::interpolation::LINEAR"
        LOWER "cudf::interpolation::LOWER"
        HIGHER "cudf::interpolation::HIGHER"
        MIDPOINT "cudf::interpolation::MIDPOINT"
        NEAREST "cudf::interpolation::NEAREST"

    # A Hack to let cython compile with __int128_t symbol
    # https://stackoverflow.com/a/27609033
    ctypedef int int128 "__int128_t"
