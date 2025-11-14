# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource

cdef extern from "cudf/strings/reverse.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] reverse(
        column_view source_strings,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
