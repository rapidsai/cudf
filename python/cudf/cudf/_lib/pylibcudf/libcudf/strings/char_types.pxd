# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar


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

cdef extern from "cudf/strings/char_types/char_types.hpp" \
        namespace "cudf::strings" nogil:

    cdef unique_ptr[column] all_characters_of_type(
        column_view source_strings,
        string_character_types types,
        string_character_types verify_types) except +

    cdef unique_ptr[column] filter_characters_of_type(
        column_view source_strings,
        string_character_types types_to_remove,
        string_scalar replacement,
        string_character_types types_to_keep) except +
