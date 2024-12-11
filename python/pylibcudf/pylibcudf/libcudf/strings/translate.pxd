# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport char_utf8


cdef extern from "cudf/strings/translate.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] translate(
        column_view input,
        vector[pair[char_utf8, char_utf8]] chars_table
    ) except +libcudf_exception_handler

    cpdef enum class filter_type(bool):
        KEEP
        REMOVE

    cdef unique_ptr[column] filter_characters(
        column_view input,
        vector[pair[char_utf8, char_utf8]] characters_to_filter,
        filter_type keep_characters,
        string_scalar replacement) except +libcudf_exception_handler
