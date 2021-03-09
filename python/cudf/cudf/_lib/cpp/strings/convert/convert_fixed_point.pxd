# Copyright (c) 2021, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport data_type

from libcpp.memory cimport unique_ptr

cdef extern from "cudf/strings/convert/convert_fixed_point.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_fixed_point(
        column_view input_col,
        data_type output_type) except +

    cdef unique_ptr[column] from_fixed_point(
        column_view input_col) except +

    cdef unique_ptr[column] is_fixed_point(
        column_view source_strings,
        data_type output_type
    ) except +
