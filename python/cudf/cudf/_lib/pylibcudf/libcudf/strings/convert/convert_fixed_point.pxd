# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.types cimport data_type


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
