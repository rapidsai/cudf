# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.regex_program cimport regex_program


cdef extern from "cudf/strings/contains.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] contains_re(
        column_view source_strings,
        regex_program) except +libcudf_exception_handler

    cdef unique_ptr[column] count_re(
        column_view source_strings,
        regex_program) except +libcudf_exception_handler

    cdef unique_ptr[column] matches_re(
        column_view source_strings,
        regex_program) except +libcudf_exception_handler

    cdef unique_ptr[column] like(
        column_view source_strings,
        string_scalar pattern,
        string_scalar escape_character) except +libcudf_exception_handler

    cdef unique_ptr[column] like(
        column_view source_strings,
        column_view patterns,
        string_scalar escape_character) except +libcudf_exception_handler
