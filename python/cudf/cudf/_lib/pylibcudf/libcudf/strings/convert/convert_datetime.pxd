# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/strings/convert/convert_datetime.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_timestamps(
        column_view input_col,
        data_type timestamp_type,
        string format) except +

    cdef unique_ptr[column] from_timestamps(
        column_view input_col,
        string format,
        column_view input_strings_names) except +

    cdef unique_ptr[column] is_timestamp(
        column_view input_col,
        string format) except +
