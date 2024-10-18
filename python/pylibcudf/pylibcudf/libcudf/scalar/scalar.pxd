# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.fixed_point.fixed_point cimport scale_type
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/scalar/scalar.hpp" namespace "cudf" nogil:
    cdef cppclass scalar:
        scalar() except +libcudf_exception_handler
        scalar(scalar other) except +libcudf_exception_handler
        data_type type() except +libcudf_exception_handler
        void set_valid_async(bool is_valid) except +libcudf_exception_handler
        bool is_valid() except +libcudf_exception_handler

    cdef cppclass numeric_scalar[T](scalar):
        numeric_scalar() except +libcudf_exception_handler
        numeric_scalar(numeric_scalar other) except +libcudf_exception_handler
        numeric_scalar(T value) except +libcudf_exception_handler
        numeric_scalar(T value, bool is_valid) except +libcudf_exception_handler
        void set_value(T value) except +libcudf_exception_handler
        T value() except +libcudf_exception_handler

    cdef cppclass timestamp_scalar[T](scalar):
        timestamp_scalar() except +libcudf_exception_handler
        timestamp_scalar(timestamp_scalar other) except +libcudf_exception_handler
        timestamp_scalar(int64_t value) except +libcudf_exception_handler
        timestamp_scalar(int64_t value, bool is_valid) except +libcudf_exception_handler
        timestamp_scalar(int32_t value) except +libcudf_exception_handler
        timestamp_scalar(int32_t value, bool is_valid) except +libcudf_exception_handler
        int64_t ticks_since_epoch_64 "ticks_since_epoch"()\
            except +libcudf_exception_handler
        int32_t ticks_since_epoch_32 "ticks_since_epoch"()\
            except +libcudf_exception_handler
        T value() except +libcudf_exception_handler

    cdef cppclass duration_scalar[T](scalar):
        duration_scalar() except +libcudf_exception_handler
        duration_scalar(duration_scalar other) except +libcudf_exception_handler
        duration_scalar(int64_t value) except +libcudf_exception_handler
        duration_scalar(int64_t value, bool is_valid) except +libcudf_exception_handler
        duration_scalar(int32_t value) except +libcudf_exception_handler
        duration_scalar(int32_t value, bool is_valid) except +libcudf_exception_handler
        int64_t ticks "count"() except +libcudf_exception_handler
        T value() except +libcudf_exception_handler

    cdef cppclass string_scalar(scalar):
        string_scalar() except +libcudf_exception_handler
        string_scalar(string st) except +libcudf_exception_handler
        string_scalar(string st, bool is_valid) except +libcudf_exception_handler
        string_scalar(string_scalar other) except +libcudf_exception_handler
        string to_string() except +libcudf_exception_handler

    cdef cppclass fixed_point_scalar[T](scalar):
        fixed_point_scalar() except +libcudf_exception_handler
        fixed_point_scalar(int64_t value,
                           scale_type scale,
                           bool is_valid) except +libcudf_exception_handler
        fixed_point_scalar(data_type value,
                           scale_type scale,
                           bool is_valid) except +libcudf_exception_handler
        int64_t value() except +libcudf_exception_handler
        # TODO: Figure out how to add an int32 overload of value()

    cdef cppclass list_scalar(scalar):
        list_scalar(column_view col) except +libcudf_exception_handler
        list_scalar(column_view col, bool is_valid) except +libcudf_exception_handler
        column_view view() except +libcudf_exception_handler

    cdef cppclass struct_scalar(scalar):
        struct_scalar(table_view cols, bool valid) except +libcudf_exception_handler
        table_view view() except +libcudf_exception_handler
