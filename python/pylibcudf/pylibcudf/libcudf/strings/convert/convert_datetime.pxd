# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/strings/convert/convert_datetime.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_timestamps(
        column_view input,
        data_type timestamp_type,
        string format) except +

    cdef unique_ptr[column] from_timestamps(
        column_view timestamps,
        string format,
        column_view names) except +

    cdef unique_ptr[column] is_timestamp(
        column_view input_col,
        string format) except +
