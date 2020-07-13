# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport data_type

from libcpp.memory cimport unique_ptr

cdef extern from "cudf/strings/convert/convert_integers.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_integers(
        column_view input_col,
        data_type output_type) except +

    cdef unique_ptr[column] from_integers(
        column_view input_col) except +

    cdef unique_ptr[column] hex_to_integers(
        column_view input_col,
        data_type output_type) except +

    cdef unique_ptr[column] is_hex(
        column_view source_strings
    ) except +
