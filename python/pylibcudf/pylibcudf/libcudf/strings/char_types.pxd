# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar


cdef extern from "cudf/strings/char_types/char_types.hpp" \
        namespace "cudf::strings" nogil:

    cpdef enum class string_character_types(uint32_t):
        DECIMAL
        NUMERIC
        DIGIT
        ALPHA
        SPACE
        UPPER
        LOWER
        ALPHANUM
        CASE_TYPES
        ALL_TYPES

    cdef unique_ptr[column] all_characters_of_type(
        column_view source_strings,
        string_character_types types,
        string_character_types verify_types) except +libcudf_exception_handler

    cdef unique_ptr[column] filter_characters_of_type(
        column_view source_strings,
        string_character_types types_to_remove,
        string_scalar replacement,
        string_character_types types_to_keep) except +libcudf_exception_handler
