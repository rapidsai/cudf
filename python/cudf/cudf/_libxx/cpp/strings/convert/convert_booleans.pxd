# Copyright (c) 2020, NVIDIA CORPORATION.
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.scalar.scalar cimport string_scalar

from libcpp.memory cimport unique_ptr

cdef extern from "cudf/strings/convert/convert_booleans.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_booleans(
        column_view input_col,
        string_scalar true_string) except +

    cdef unique_ptr[column] from_booleans(
        column_view input_col,
        string_scalar true_string,
        string_scalar false_string) except +
