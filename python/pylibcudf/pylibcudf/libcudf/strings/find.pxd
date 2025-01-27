# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/find.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] contains(
        column_view source_strings,
        string_scalar target) except +libcudf_exception_handler

    cdef unique_ptr[column] contains(
        column_view source_strings,
        column_view target_strings) except +libcudf_exception_handler

    cdef unique_ptr[column] ends_with(
        column_view source_strings,
        string_scalar target) except +libcudf_exception_handler

    cdef unique_ptr[column] ends_with(
        column_view source_strings,
        column_view target_strings) except +libcudf_exception_handler

    cdef unique_ptr[column] starts_with(
        column_view source_strings,
        string_scalar target) except +libcudf_exception_handler

    cdef unique_ptr[column] starts_with(
        column_view source_strings,
        column_view target_strings) except +libcudf_exception_handler

    cdef unique_ptr[column] find(
        column_view source_strings,
        string_scalar target,
        size_type start,
        size_type stop) except +libcudf_exception_handler

    cdef unique_ptr[column] find(
        column_view source_strings,
        column_view target,
        size_type start) except +libcudf_exception_handler

    cdef unique_ptr[column] rfind(
        column_view source_strings,
        string_scalar target,
        size_type start,
        size_type stop) except +libcudf_exception_handler
