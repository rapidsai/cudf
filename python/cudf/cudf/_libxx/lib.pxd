# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import numpy as np

from libc.stdint cimport int32_t, uint32_t, int64_t
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.pair cimport pair
from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer


cdef extern from "cudf/types.hpp" namespace "cudf" nogil:
    ctypedef int32_t size_type
    ctypedef uint32_t bitmask_type

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

    ctypedef enum interpolation:
        LINEAR "cudf::experimental::interpolation::LINEAR"
        LOWER "cudf::experimental::interpolation::LOWER"
        HIGHER "cudf::experimental::interpolation::HIGHER"
        MIDPOINT "cudf::experimental::interpolation::MIDPOINT"
        NEAREST "cudf::experimental::interpolation::NEAREST"

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

cdef extern from "cudf/aggregation.hpp" namespace "cudf::experimental" nogil:
    cdef cppclass aggregation:
        pass

cdef extern from "cudf/wrappers/bool.hpp" namespace "cudf::experimental" nogil:
    ctypedef bool bool8

cdef extern from "cudf/wrappers/timestamps.hpp" namespace "cudf" nogil:
    ctypedef int64_t timestamp_s
    ctypedef int64_t timestamp_ms
    ctypedef int64_t timestamp_us
    ctypedef int64_t timestamp_ns

cdef extern from "cudf/scalar/scalar.hpp" namespace "cudf" nogil:
    cdef cppclass scalar:
        scalar() except +
        scalar(scalar other) except +
        data_type type() except +
        void set_valid(bool is_valid) except +
        bool is_valid() except +

    cdef cppclass numeric_scalar[T](scalar):
        numeric_scalar() except +
        numeric_scalar(numeric_scalar other) except +
        numeric_scalar(T value) except +
        numeric_scalar(T value, bool is_valid) except +
        void set_value(T value) except +
        T value() except +

    cdef cppclass timestamp_scalar[T](scalar):
        timestamp_scalar() except +
        timestamp_scalar(timestamp_scalar other) except +
        timestamp_scalar(int64_t value) except +
        timestamp_scalar(int64_t value, bool is_valid) except +
        timestamp_scalar(int32_t value) except +
        timestamp_scalar(int32_t value, bool is_valid) except +
        int64_t ticks_since_epoch_64 "ticks_since_epoch"() except +
        int32_t ticks_since_epoch_32 "ticks_since_epoch"() except +
        T value() except +

    cdef cppclass string_scalar(scalar):
        string_scalar() except +
        string_scalar(string st) except +
        string_scalar(string st, bool is_valid) except +
        string_scalar(string_scalar other) except +
        string to_string() except +

cdef extern from "cudf/scalar/scalar_factories.hpp" namespace "cudf" nogil:
    cdef unique_ptr[scalar] make_numeric_scalar(data_type dtype) except +

# Note: declaring `move()` with `except +` doesn't work.
#
# Consider:
#     cdef unique_ptr[int] x = move(y)
#
# If `move()` is declared with `except +`, the generated C++ code
# looks something like this:
#
#    std::unique_ptr<int>  __pyx_v_x;
#    std::unique_ptr<int>  __pyx_v_y;
#    std::unique_ptr<int>  __pyx_t_1;
#    try {
#      __pyx_t_1 = std::move(__pyx_v_y);
#    } catch(...) {
#      __Pyx_CppExn2PyErr();
#      __PYX_ERR(0, 8, __pyx_L1_error)
#    }
#    __pyx_v_x = __pyx_t_1;
#
# where the last statement will result in a compiler error
# (copying a unique_ptr).
#
