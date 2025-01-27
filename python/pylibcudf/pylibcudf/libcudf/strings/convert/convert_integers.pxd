# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/strings/convert/convert_integers.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_integers(
        column_view input,
        data_type output_type) except +libcudf_exception_handler

    cdef unique_ptr[column] from_integers(
        column_view integers) except +libcudf_exception_handler

    cdef unique_ptr[column] is_integer(
        column_view input
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] is_integer(
        column_view input,
        data_type int_type
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] hex_to_integers(
        column_view input,
        data_type output_type) except +

    cdef unique_ptr[column] is_hex(
        column_view input
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] integers_to_hex(
        column_view input) except +libcudf_exception_handler
