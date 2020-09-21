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

    ctypedef enum include_nulls "cudf::include_nulls":
        NO "cudf::include_nulls::NO"
        YES "cudf::include_nulls::YES"

    ctypedef enum null_order "cudf::null_order":
        AFTER "cudf::null_order::AFTER"
        BEFORE "cudf::null_order::BEFORE"

    ctypedef enum sorted "cudf::sorted":
        NO "cudf::sorted::NO"
        YES "cudf::sorted::YES"

    ctypedef enum include_nulls "cudf::include_nulls":
        NO "cudf::include_nulls::NO"
        YES "cudf::include_nulls::YES"

    cdef enum type_id:
        EMPTY = 0
        INT8 = 1
        INT16 = 2
        INT32 = 3
        INT64 = 4
        FLOAT32 = 5
        FLOAT64 = 6
        BOOL8 = 7
        TIMESTAMP_DAYS = 8
        TIMESTAMP_SECONDS = 9
        TIMESTAMP_MILLISECONDS = 10
        TIMESTAMP_MICROSECONDS = 11
        TIMESTAMP_NANOSECONDS = 12
        DICTIONARY32 = 13
        STRING = 14
        NUM_TYPE_IDS = 15

    cdef cppclass data_type:
        data_type() except +
        data_type(const data_type&) except +
        data_type(type_id id) except +
        type_id id() except +

cdef extern from "cudf/types.hpp" namespace "cudf::experimental" nogil:
    ctypedef enum interpolation:
        LINEAR "cudf::experimental::interpolation::LINEAR"
        LOWER "cudf::experimental::interpolation::LOWER"
        HIGHER "cudf::experimental::interpolation::HIGHER"
        MIDPOINT "cudf::experimental::interpolation::MIDPOINT"
        NEAREST "cudf::experimental::interpolation::NEAREST"
