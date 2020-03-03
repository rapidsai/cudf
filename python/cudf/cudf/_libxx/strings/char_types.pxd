# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.column.column cimport column

cdef extern from "cudf/strings/char_types/char_types.hpp" \
        namespace "cudf::strings" nogil:

    ctypedef enum string_character_types:
        DECIMAL 'cudf::strings::string_character_types::DECIMAL' = 1 << 0
        NUMERIC  'cudf::strings::string_character_types::NUMERIC' = 1 << 1
        DIGIT 'cudf::strings::string_character_types::DIGIT' = 1 << 2
        ALPHA 'cudf::strings::string_character_types::ALPHA' = 1 << 3
        SPACE 'cudf::strings::string_character_types::SPACE' = 1 << 4
        UPPER 'cudf::strings::string_character_types::UPPER' = 1 << 5
        LOWER 'cudf::strings::string_character_types::LOWER' = 1 << 6
        ALPHANUM 'cudf::strings::string_character_types::ALPHANUM' \
            = DECIMAL | NUMERIC | DIGIT | ALPHA

cdef extern from "cudf/strings/char_types/char_types.hpp" \
        namespace "cudf::strings" nogil:

    cdef unique_ptr[column] all_characters_of_type(
        column_view source_strings,
        string_character_types types) except +
