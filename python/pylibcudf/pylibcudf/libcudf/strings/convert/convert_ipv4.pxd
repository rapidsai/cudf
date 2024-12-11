# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "cudf/strings/convert/convert_ipv4.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] ipv4_to_integers(
        column_view input) except +libcudf_exception_handler

    cdef unique_ptr[column] integers_to_ipv4(
        column_view integers) except +libcudf_exception_handler

    cdef unique_ptr[column] is_ipv4(
        column_view input
    ) except +libcudf_exception_handler
