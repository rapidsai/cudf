# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column slice_strings(
    Column input,
    ColumnOrScalar start=*,
    ColumnOrScalar stop=*,
    Scalar step=*
)
