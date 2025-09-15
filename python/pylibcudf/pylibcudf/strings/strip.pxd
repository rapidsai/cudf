# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.side_type cimport side_type
from rmm.pylibrmm.stream cimport Stream


cpdef Column strip(
    Column input,
    side_type side=*,
    Scalar to_strip=*,
    Stream stream=*
)
