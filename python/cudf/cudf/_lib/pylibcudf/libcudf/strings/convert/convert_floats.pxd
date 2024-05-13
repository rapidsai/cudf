# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport data_type


cdef extern from "cudf/strings/convert/convert_floats.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_floats(
        column_view input_col,
        data_type output_type) except +

    cdef unique_ptr[column] from_floats(
        column_view input_col) except +

    cdef unique_ptr[column] is_float(
        column_view source_strings
    ) except +
