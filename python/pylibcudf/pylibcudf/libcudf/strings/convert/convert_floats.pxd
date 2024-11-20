# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/strings/convert/convert_floats.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_floats(
        column_view strings,
        data_type output_type) except +libcudf_exception_handler

    cdef unique_ptr[column] from_floats(
        column_view floats) except +libcudf_exception_handler

    cdef unique_ptr[column] is_float(
        column_view input
    ) except +libcudf_exception_handler
