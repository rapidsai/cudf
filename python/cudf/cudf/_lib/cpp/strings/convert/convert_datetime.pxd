# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport data_type

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

cdef extern from "cudf/strings/convert/convert_datetime.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_timestamps(
        column_view input_col,
        data_type timestamp_type,
        string format) except +

    cdef unique_ptr[column] from_timestamps(
        column_view input_col,
        string format) except +
