# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/strings/convert/convert_durations.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_durations(
        const column_view & input,
        data_type duration_type,
        const string & format,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] from_durations(
        const column_view & durations,
        const string & format,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler
