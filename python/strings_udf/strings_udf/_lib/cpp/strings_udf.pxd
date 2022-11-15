# Copyright (c) 2022, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type
from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer


cdef extern from "cudf/strings/udf/udf_string.hpp" namespace \
        "cudf::strings::udf" nogil:
    cdef cppclass udf_string

cdef extern from "cudf/strings/udf/udf_apis.hpp"  namespace \
        "cudf::strings::udf" nogil:
    cdef unique_ptr[device_buffer] to_string_view_array(column_view) except +
    cdef unique_ptr[column] column_from_udf_string_array(
        udf_string* strings, size_type size,
    ) except +
    cdef void free_udf_string_array(
        udf_string* strings, size_type size
    ) except +

cdef extern from "cudf/strings/detail/char_tables.hpp" namespace \
        "cudf::strings::detail" nogil:
    cdef const uint8_t* get_character_flags_table() except +
