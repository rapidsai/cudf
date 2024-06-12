# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.scalar cimport Scalar

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column slice_strings(
    Column input,
    ColumnOrScalar start=*,
    ColumnOrScalar stop=*,
    Scalar step=*
)
