# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.column.column cimport column

cdef extern from "cudf/strings/char_types/char_types.hpp" \
        namespace "cudf::strings" nogil:

    ctypedef enum string_character_types:
        DECIMAL = 1 << 0                              # binary 00000001
        NUMERIC = 1 << 1                              # binary 00000010
        DIGIT = 1 << 2                                # binary 00000100
        ALPHA = 1 << 3                                # binary 00001000
        SPACE = 1 << 4                                # binary 00010000
        UPPER = 1 << 5                                # binary 00100000
        LOWER = 1 << 6                                # binary 01000000
        ALPHANUM = DECIMAL | NUMERIC | DIGIT | ALPHA  # binary 00001111

    cdef unique_ptr[column] all_characters_of_type(
        column_view source_strings,
        string_character_types types) except +
