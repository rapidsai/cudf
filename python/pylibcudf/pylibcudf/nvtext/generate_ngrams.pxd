# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.stream cimport Stream


cpdef Column generate_ngrams(
    Column input, size_type ngrams, Scalar separator, Stream stream=*
)

cpdef Column generate_character_ngrams(
    Column input, size_type ngrams=*, Stream stream=*
)

cpdef Column hash_character_ngrams(
    Column input, size_type ngrams, uint32_t seed, Stream stream=*
)
