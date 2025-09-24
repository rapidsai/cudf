# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.stream cimport Stream

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column find(
    Column input,
    ColumnOrScalar target,
    size_type start=*,
    size_type stop=*,
    Stream stream=*
)

cpdef Column rfind(
    Column input,
    Scalar target,
    size_type start=*,
    size_type stop=*,
    Stream stream=*
)

cpdef Column contains(
    Column input,
    ColumnOrScalar target,
    Stream stream=*
)

cpdef Column starts_with(
    Column input,
    ColumnOrScalar target,
    Stream stream=*
)

cpdef Column ends_with(
    Column input,
    ColumnOrScalar target,
    Stream stream=*
)
