# Copyright (c) 2019-2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np

from libc.stdint cimport int32_t, uint32_t, int64_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.pair cimport pair

from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer, move


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

cdef extern from "cudf/column/column.hpp" namespace "cudf" nogil:
    cdef cppclass column_contents "cudf::column::contents":
        unique_ptr[device_buffer] data
        unique_ptr[device_buffer] null_mask
        vector[unique_ptr[column]] children

    cdef cppclass column:
        column() except +
        column(const column& other) except +
        column(data_type dtype, size_type size, device_buffer&& data) except +
        size_type size() except +
        bool has_nulls() except +
        data_type type() except +
        column_view view() except +
        mutable_column_view mutable_view() except +
        column_contents release() except +

cdef extern from "cudf/column/column_view.hpp" namespace "cudf" nogil:
    cdef cppclass column_view:
        column_view() except +
        column_view(const column_view& other) except +

        column_view& operator=(const column_view&) except +
        column_view& operator=(column_view&&) except +
        column_view(data_type type, size_type size, const void* data) except +
        column_view(data_type type, size_type size, const void* data,
                    const bitmask_type* null_mask) except +
        column_view(data_type type, size_type size, const void* data,
                    const bitmask_type* null_mask, size_type null_count) except +
        column_view(data_type type, size_type size, const void* data,
                    const bitmask_type* null_mask, size_type null_count,
                    size_type offset) except +
        column_view(data_type type, size_type size, const void* data,
                    const bitmask_type* null_mask, size_type null_count,
                    size_type offset, vector[column_view] children) except +
        const T* data[T]() except +
        const T* head[T]() except +
        const bitmask_type* null_mask() except +
        size_type size() except +
        data_type type() except +
        bool nullable() except +
        size_type null_count() except +
        bool has_nulls() except +
        size_type offset() except +
        size_type num_children() except +
        column_view child(size_type) except +

    cdef cppclass mutable_column_view:
        mutable_column_view() except +
        mutable_column_view(const mutable_column_view&) except +
        mutable_column_view& operator=(const mutable_column_view&) except +
        mutable_column_view(data_type type, size_type size, const void* data) except +
        mutable_column_view(data_type type, size_type size, const void* data,
                            const bitmask_type* null_mask) except +
        mutable_column_view(
            data_type type, size_type size, const void* data,
            const bitmask_type* null_mask, size_type null_count
        ) except +
        mutable_column_view(
            data_type type, size_type size, const void* data,
            const bitmask_type* null_mask, size_type null_count,
            size_type offset
        ) except +
        mutable_column_view(
            data_type type, size_type size, const void* data,
            const bitmask_type* null_mask, size_type null_count,
            size_type offset, vector[mutable_column_view] children
        ) except +
        T* data[T]() except +
        T* head[T]() except +
        bitmask_type* null_mask() except +
        size_type size() except +
        data_type type() except +
        bool nullable() except +
        size_type null_count() except +
        bool has_nulls() except +
        size_type offset() except +
        size_type num_children() except +
        mutable_column_view& child(size_type) except +

cdef extern from "cudf/table/table_view.hpp" namespace "cudf" nogil:
    cdef cppclass table_view:
        table_view() except +
        table_view(const vector[column_view]) except +
        column_view column(size_type column_index) except +
        size_type num_columns() except +
        size_type num_rows() except +

    cdef cppclass mutable_table_view:
        mutable_table_view() except +
        mutable_table_view(const vector[mutable_column_view]) except +
        mutable_column_view column(size_type column_index) except +
        size_type num_columns() except +
        size_type num_rows() except +

cdef extern from "cudf/table/table.hpp" namespace "cudf::experimental" nogil:
    cdef cppclass table:
        table(const table&) except +
        table(vector[unique_ptr[column]]&& columns) except +
        table(table_view) except +
        size_type num_columns() except +
        table_view view() except +
        mutable_table_view mutable_view() except +
        vector[unique_ptr[column]] release() except +

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
#    std::unique_ptr<int>  __pyx_v_y;
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
cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[column] move(unique_ptr[column])
    cdef unique_ptr[table] move(unique_ptr[table])
    cdef unique_ptr[aggregation] move(unique_ptr[aggregation])
    cdef vector[unique_ptr[column]] move(vector[unique_ptr[column]])
    cdef device_buffer move(device_buffer)
    cdef unique_ptr[scalar] move(unique_ptr[scalar])
    cdef pair[unique_ptr[device_buffer], size_type] move(
        pair[unique_ptr[device_buffer], size_type]
    )
    cdef column_contents move(column_contents)
