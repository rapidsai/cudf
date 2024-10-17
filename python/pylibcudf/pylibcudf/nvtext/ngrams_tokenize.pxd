# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar


cpdef Column ngrams_tokenize(
    Column input,
    size_type ngrams,
    Scalar delimiter,
    Scalar separator
)
