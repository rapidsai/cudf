# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uint64_t
from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column minhash(
    Column input,
    uint32_t seed,
    Column a,
    Column b,
    size_type width
)

cpdef Column minhash64(
    Column input,
    uint64_t seed,
    Column a,
    Column b,
    size_type width
)
