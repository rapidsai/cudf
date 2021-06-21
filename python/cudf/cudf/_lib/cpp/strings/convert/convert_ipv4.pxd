# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

from libcpp.memory cimport unique_ptr

cdef extern from "cudf/strings/convert/convert_ipv4.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] ipv4_to_integers(
        column_view input_col) except +

    cdef unique_ptr[column] integers_to_ipv4(
        column_view input_col) except +

    cdef unique_ptr[column] is_ipv4(
        column_view source_strings
    ) except +
