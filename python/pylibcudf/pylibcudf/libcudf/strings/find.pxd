# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/strings/find.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] contains(
        column_view source_strings,
        string_scalar target,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] contains(
        column_view source_strings,
        column_view target_strings,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] ends_with(
        column_view source_strings,
        string_scalar target,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] ends_with(
        column_view source_strings,
        column_view target_strings,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] starts_with(
        column_view source_strings,
        string_scalar target,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] starts_with(
        column_view source_strings,
        column_view target_strings,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] find(
        column_view source_strings,
        string_scalar target,
        size_type start,
        size_type stop,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] find(
        column_view source_strings,
        column_view target,
        size_type start,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] rfind(
        column_view source_strings,
        string_scalar target,
        size_type start,
        size_type stop,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler
