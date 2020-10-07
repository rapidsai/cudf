# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, uint32_t


cdef extern from "cudf/types.hpp" namespace "cudf" nogil:
    ctypedef int32_t size_type
    ctypedef uint32_t bitmask_type
    ctypedef uint32_t char_utf8

    cdef enum:
        UNKNOWN_NULL_COUNT = -1

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

    ctypedef enum type_id "cudf::type_id":
        EMPTY "cudf::type_id::EMPTY"
        INT8  "cudf::type_id::INT8"
        INT16 "cudf::type_id::INT16"
        INT32 "cudf::type_id::INT32"
        INT64 "cudf::type_id::INT64"
        UINT8 "cudf::type_id::UINT8"
        UINT16 "cudf::type_id::UINT16"
        UINT32 "cudf::type_id::UINT32"
        UINT64 "cudf::type_id::UINT64"
        FLOAT32 "cudf::type_id::FLOAT32"
        FLOAT64 "cudf::type_id::FLOAT64"
        BOOL8 "cudf::type_id::BOOL8"
        TIMESTAMP_DAYS "cudf::type_id::TIMESTAMP_DAYS"
        TIMESTAMP_SECONDS "cudf::type_id::TIMESTAMP_SECONDS"
        TIMESTAMP_MILLISECONDS "cudf::type_id::TIMESTAMP_MILLISECONDS"
        TIMESTAMP_MICROSECONDS "cudf::type_id::TIMESTAMP_MICROSECONDS"
        TIMESTAMP_NANOSECONDS "cudf::type_id::TIMESTAMP_NANOSECONDS"
        DICTIONARY32 "cudf::type_id::DICTIONARY32"
        STRING "cudf::type_id::STRING"
        LIST "cudf::type_id::LIST"
        STRUCT "cudf::type_id::STRUCT"
        NUM_TYPE_IDS "cudf::type_id::NUM_TYPE_IDS"
        DURATION_SECONDS "cudf::type_id::DURATION_SECONDS"
        DURATION_MILLISECONDS "cudf::type_id::DURATION_MILLISECONDS"
        DURATION_MICROSECONDS "cudf::type_id::DURATION_MICROSECONDS"
        DURATION_NANOSECONDS "cudf::type_id::DURATION_NANOSECONDS"

    ctypedef enum hash_id "cudf::hash_id":
        HASH_IDENTITY "cudf::hash_id::HASH_IDENTITY"
        HASH_MURMUR3 "cudf::hash_id::HASH_MURMUR3"
        HASH_MD5 "cudf::hash_id::HASH_MD5"

    cdef cppclass data_type:
        data_type() except +
        data_type(const data_type&) except +
        data_type(type_id id) except +
        type_id id() except +

cdef extern from "cudf/types.hpp" namespace "cudf" nogil:
    ctypedef enum interpolation:
        LINEAR "cudf::interpolation::LINEAR"
        LOWER "cudf::interpolation::LOWER"
        HIGHER "cudf::interpolation::HIGHER"
        MIDPOINT "cudf::interpolation::MIDPOINT"
        NEAREST "cudf::interpolation::NEAREST"
