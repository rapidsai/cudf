# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uint64_t
from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.table cimport table
from libcpp.vector cimport vector

#from cudf._lib.cpp.hash cimport hash_id

from .column cimport Column
from .table cimport Table


cpdef Column murmurhash3_x86_32(
    Table input, 
    uint32_t seed
)

cpdef Table murmurhash3_x64_128(
    Table input,
    uint64_t seed
)

cpdef Column md5(Table input)
cpdef Column sha1(Table input)
cpdef Column sha224(Table input)
cpdef Column sha256(Table input)
cpdef Column sha384(Table input)
cpdef Column sha512(Table input)

cpdef Column xxhash_64(
    Table input,
    uint64_t seed
)
