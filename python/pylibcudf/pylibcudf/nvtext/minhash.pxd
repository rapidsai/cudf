# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uint64_t
from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column minhash(Column input, ColumnOrScalar seeds, size_type width=*)

cpdef Column minhash_permuted(
    Column input,
    uint32_t seed,
    Column a,
    Column b,
    size_type width
)

cpdef Column minhash64(Column input, ColumnOrScalar seeds, size_type width=*)

cpdef Column minhash64_permuted(
    Column input,
    uint64_t seed,
    Column a,
    Column b,
    size_type width
)

cpdef Column word_minhash(Column input, Column seeds)

cpdef Column word_minhash64(Column input, Column seeds)
