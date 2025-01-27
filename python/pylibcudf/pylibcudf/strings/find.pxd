# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar

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
