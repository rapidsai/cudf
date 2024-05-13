# Copyright (c) 2022-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t, uint16_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.types cimport size_type


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
    cdef const uint16_t* get_character_cases_table() except +
    cdef const void* get_special_case_mapping_table() except +
