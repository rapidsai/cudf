# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/strings/convert/convert_booleans.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_booleans(
        column_view input,
        string_scalar true_string,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] from_booleans(
        column_view booleans,
        string_scalar true_string,
        string_scalar false_string,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler
