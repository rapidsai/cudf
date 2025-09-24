# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.stream cimport Stream


cpdef Column ngrams_tokenize(
    Column input,
    size_type ngrams,
    Scalar delimiter,
    Scalar separator,
    Stream stream=*
)
