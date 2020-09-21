# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.column.column cimport column

cdef extern from "cudf/strings/char_types/char_types.hpp" \
        namespace "cudf::strings" nogil:

    ctypedef enum string_character_types:
        DECIMAL 'cudf::strings::string_character_types::DECIMAL'
        NUMERIC  'cudf::strings::string_character_types::NUMERIC'
        DIGIT 'cudf::strings::string_character_types::DIGIT'
        ALPHA 'cudf::strings::string_character_types::ALPHA'
        SPACE 'cudf::strings::string_character_types::SPACE'
        UPPER 'cudf::strings::string_character_types::UPPER'
        LOWER 'cudf::strings::string_character_types::LOWER'
        ALPHANUM 'cudf::strings::string_character_types::ALPHANUM'

cdef extern from "cudf/strings/char_types/char_types.hpp" \
        namespace "cudf::strings" nogil:

    cdef unique_ptr[column] all_characters_of_type(
        column_view source_strings,
        string_character_types types) except +
