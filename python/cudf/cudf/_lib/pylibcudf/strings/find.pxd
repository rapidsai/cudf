# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.pylibcudf.scalar cimport Scalar

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column find(
    Column input,
    ColumnOrScalar target,
    size_type start=*,
    size_type stop=*
)

cpdef Column rfind(
    Column input,
    Scalar target,
    size_type start=*,
    size_type stop=*
)

cpdef Column contains(
    Column input,
    ColumnOrScalar target,
)

cpdef Column starts_with(
    Column input,
    ColumnOrScalar target,
)

cpdef Column ends_with(
    Column input,
    ColumnOrScalar target,
)
