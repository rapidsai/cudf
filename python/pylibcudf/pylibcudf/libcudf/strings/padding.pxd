# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.side_type cimport side_type
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/strings/padding.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] pad(
        column_view input,
        size_type width,
        side_type side,
        string fill_char,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] zfill(
        column_view input,
        size_type width,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] zfill_by_widths(
        column_view input,
        column_view widths,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler
