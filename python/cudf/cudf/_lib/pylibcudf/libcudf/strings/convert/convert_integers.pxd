# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/strings/convert/convert_integers.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_integers(
        column_view input_col,
        data_type output_type) except +

    cdef unique_ptr[column] from_integers(
        column_view input_col) except +

    cdef unique_ptr[column] is_integer(
        column_view source_strings
    ) except +

    cdef unique_ptr[column] hex_to_integers(
        column_view input_col,
        data_type output_type) except +

    cdef unique_ptr[column] is_hex(
        column_view source_strings
    ) except +

    cdef unique_ptr[column] integers_to_hex(
        column_view input_col) except +
