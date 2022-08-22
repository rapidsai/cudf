# Copyright (c) 2021-2022, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type
from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer


cdef extern from "udf_apis.hpp":
    cdef cppclass udf_module
    cdef unique_ptr[udf_module] create_udf_module(string, vector[string])
    cdef unique_ptr[column] call_udf(
        udf_module, string, size_type, vector[column_view])
    cdef unique_ptr[device_buffer] to_string_view_array(column_view)
    cdef unique_ptr[column] from_dstring_array(void*, size_t)

cdef extern from "cudf/strings/detail/char_tables.hpp" namespace \
        "cudf::strings::detail":
    cdef const uint8_t* get_character_flags_table() except +
