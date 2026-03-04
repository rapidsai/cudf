# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport numeric_scalar
from pylibcudf.libcudf.types cimport size_type

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/strings/slice.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] slice_strings(
        column_view source_strings,
        numeric_scalar[size_type] start,
        numeric_scalar[size_type] end,
        numeric_scalar[size_type] step,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] slice_strings(
        column_view source_strings,
        column_view starts,
        column_view stops,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
