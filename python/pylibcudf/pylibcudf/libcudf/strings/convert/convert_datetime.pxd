# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type

from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "cudf/strings/convert/convert_datetime.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_timestamps(
        column_view input,
        data_type timestamp_type,
        string format,
        cuda_stream_view stream) except +libcudf_exception_handler

    cdef unique_ptr[column] from_timestamps(
        column_view timestamps,
        string format,
        column_view names,
        cuda_stream_view stream) except +libcudf_exception_handler

    cdef unique_ptr[column] is_timestamp(
        column_view input_col,
        string format,
        cuda_stream_view stream) except +libcudf_exception_handler
