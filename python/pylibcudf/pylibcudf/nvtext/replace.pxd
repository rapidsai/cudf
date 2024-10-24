# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar


cpdef Column replace_tokens(
    Column input,
    Column targets,
    Column replacements,
    Scalar delimiter=*,
)

cpdef Column filter_tokens(
    Column input,
    size_type min_token_length,
    Scalar replacement=*,
    Scalar delimiter=*
)
