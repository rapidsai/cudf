# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.side_type cimport side_type


cpdef Column strip(
    Column input,
    side_type side=*,
    Scalar to_strip=*
)
