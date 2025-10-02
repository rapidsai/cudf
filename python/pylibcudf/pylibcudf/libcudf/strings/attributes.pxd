# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/strings/attributes.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] count_characters(
        column_view source_strings,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] count_bytes(
        column_view source_strings,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] code_points(
        column_view source_strings,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler
