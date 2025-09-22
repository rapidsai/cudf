# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uint64_t
from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.stream cimport Stream

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column minhash(
    Column input,
    uint32_t seed,
    Column a,
    Column b,
    size_type width,
    Stream stream=*
)

cpdef Column minhash64(
    Column input,
    uint64_t seed,
    Column a,
    Column b,
    size_type width,
    Stream stream=*
)

cpdef Column minhash_ngrams(
    Column input,
    size_type width,
    uint32_t seed,
    Column a,
    Column b,
    Stream stream=*
)

cpdef Column minhash64_ngrams(
    Column input,
    size_type width,
    uint64_t seed,
    Column a,
    Column b,
    Stream stream=*
)
