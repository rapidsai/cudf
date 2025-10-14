# Copyright (c) 2021-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/strings/repeat_strings.hpp" namespace "cudf::strings" \
        nogil:

    cdef unique_ptr[column] repeat_strings(
        column_view input,
        size_type repeat_times,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] repeat_strings(
        column_view input,
        column_view repeat_times,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
