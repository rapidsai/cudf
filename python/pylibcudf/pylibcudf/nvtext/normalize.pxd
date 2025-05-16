# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.normalize cimport character_normalizer

cdef class CharacterNormalizer:
    cdef unique_ptr[character_normalizer] c_obj

cpdef Column normalize_spaces(Column input)

cpdef Column characters_normalize(Column input, bool do_lower_case)

cpdef Column normalize_characters(
  Column input,
  CharacterNormalizer normalizer
)
