# Copyright (c) 2021-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type

from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "cudf/strings/convert/convert_fixed_point.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_fixed_point(
        column_view input,
        data_type output_type,
        cuda_stream_view stream) except +libcudf_exception_handler

    cdef unique_ptr[column] from_fixed_point(
        column_view input,
        cuda_stream_view stream) except +libcudf_exception_handler

    cdef unique_ptr[column] is_fixed_point(
        column_view input,
        data_type decimal_type,
        cuda_stream_view stream
    ) except +libcudf_exception_handler
