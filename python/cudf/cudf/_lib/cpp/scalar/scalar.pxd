# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport (
    int32_t, int64_t
)
from libcpp cimport bool
from libcpp.string cimport string

from cudf._lib.cpp.types cimport data_type

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

    cdef cppclass duration_scalar[T](scalar):
        duration_scalar() except +
        duration_scalar(duration_scalar other) except +
        duration_scalar(int64_t value) except +
        duration_scalar(int64_t value, bool is_valid) except +
        duration_scalar(int32_t value) except +
        duration_scalar(int32_t value, bool is_valid) except +
        int64_t ticks "count"() except +
        T value() except +

    cdef cppclass string_scalar(scalar):
        string_scalar() except +
        string_scalar(string st) except +
        string_scalar(string st, bool is_valid) except +
        string_scalar(string_scalar other) except +
        string to_string() except +
