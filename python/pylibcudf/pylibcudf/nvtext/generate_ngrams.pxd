# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar


cpdef Column generate_ngrams(Column input, size_type ngrams, Scalar separator)

cpdef Column generate_character_ngrams(Column input, size_type ngrams=*)

cpdef Column hash_character_ngrams(Column input, size_type ngrams=*)
