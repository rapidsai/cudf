# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport column_view

from libcpp.memory cimport unique_ptr

cdef extern from "cudf/strings/convert/convert_urls.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] url_encode(
        column_view input_col) except +

    cdef unique_ptr[column] url_decode(
        column_view input_col) except +
