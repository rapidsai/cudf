# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/strings/convert/convert_fixed_point.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_fixed_point(
        column_view input,
        data_type output_type) except +

    cdef unique_ptr[column] from_fixed_point(
        column_view input) except +

    cdef unique_ptr[column] is_fixed_point(
        column_view input,
        data_type decimal_type
    ) except +
