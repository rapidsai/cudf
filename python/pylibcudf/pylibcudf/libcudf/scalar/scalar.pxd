# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.fixed_point.fixed_point cimport scale_type
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport data_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/scalar/scalar.hpp" namespace "cudf" nogil:
    cdef cppclass scalar:
        scalar() except +libcudf_exception_handler
        scalar(scalar other) except +libcudf_exception_handler
        data_type type() except +libcudf_exception_handler
        void set_valid_async(bool is_valid) except +libcudf_exception_handler
        bool is_valid() except +libcudf_exception_handler

    cdef cppclass numeric_scalar[T](scalar):
        void set_value(T value) except +libcudf_exception_handler
        T value() except +libcudf_exception_handler

    cdef cppclass timestamp_scalar[T](scalar):
        void set_value(T value) except +libcudf_exception_handler

    cdef cppclass duration_scalar[T](scalar):
        void set_value(T value) except +libcudf_exception_handler

    cdef cppclass string_scalar(scalar):
        string to_string() except +libcudf_exception_handler

    cdef cppclass fixed_point_scalar[T](scalar):
        T value() except +libcudf_exception_handler

    cdef cppclass list_scalar(scalar):
        pass

    cdef cppclass struct_scalar(scalar):
        pass
