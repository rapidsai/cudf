# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uint8_t, uint16_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type

from rmm.librmm.device_buffer cimport device_buffer

cdef extern from "cudf/strings/udf/managed_udf_string.cuh" namespace \
        "cudf::strings::udf" nogil:
    cdef cppclass managed_udf_string

cdef extern from "cudf/strings/udf/udf_apis.hpp"  namespace \
        "cudf::strings::udf" nogil:

    cdef int get_cuda_build_version() except +libcudf_exception_handler
    cdef unique_ptr[device_buffer] to_string_view_array(
        column_view
    ) except +libcudf_exception_handler
    cdef unique_ptr[column] column_from_managed_udf_string_array(
        managed_udf_string* strings, size_type size,
    ) except +libcudf_exception_handler

cdef extern from "cudf/strings/detail/char_tables.hpp" namespace \
        "cudf::strings::detail" nogil:
    cdef const uint8_t* get_character_flags_table() except +libcudf_exception_handler
    cdef const uint16_t* get_character_cases_table() except +libcudf_exception_handler
    cdef const void* get_special_case_mapping_table() except +libcudf_exception_handler
