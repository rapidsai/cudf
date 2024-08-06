# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.strings.side_type cimport side_type
from cudf._lib.pylibcudf.scalar cimport Scalar

cpdef Column strip(
    Column input,
    side_type side,
    Scalar to_strip
)
