# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uint64_t
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .table cimport Table


cpdef Column murmurhash3_x86_32(
    Table input,
    uint32_t seed=*,
    Stream stream=*
)

cpdef Table murmurhash3_x64_128(
    Table input,
    uint64_t seed=*,
    Stream stream=*
)

cpdef Column xxhash_32(
    Table input,
    uint32_t seed=*,
    Stream stream=*
)

cpdef Column xxhash_64(
    Table input,
    uint64_t seed=*,
    Stream stream=*
)

cpdef Column md5(Table input, Stream stream=*)
cpdef Column sha1(Table input, Stream stream=*)
cpdef Column sha224(Table input, Stream stream=*)
cpdef Column sha256(Table input, Stream stream=*)
cpdef Column sha384(Table input, Stream stream=*)
cpdef Column sha512(Table input, Stream stream=*)
