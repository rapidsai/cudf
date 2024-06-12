# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "cudf/strings/attributes.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] count_characters(
        column_view source_strings) except +

    cdef unique_ptr[column] count_bytes(
        column_view source_strings) except +

    cdef unique_ptr[column] code_points(
        column_view source_strings) except +
