# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings.char_types cimport string_character_types
from pylibcudf.scalar cimport Scalar


cpdef Column all_characters_of_type(
    Column source_strings,
    string_character_types types,
    string_character_types verify_types
)

cpdef Column filter_characters_of_type(
    Column source_strings,
    string_character_types types_to_remove,
    Scalar replacement,
    string_character_types types_to_keep
)
