# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view


cdef extern from "cudf/hashing.hpp" namespace "cudf::hashing" nogil:

    cdef unique_ptr[column] murmurhash3_x86_32 "cudf::hashing::murmurhash3_x86_32" (
        const table_view& input,
        const uint32_t seed
    ) except +

    cdef unique_ptr[column] md5 "cudf::hashing::md5" (
        const table_view& input
    ) except +

    cdef unique_ptr[column] xxhash_64 "cudf::hashing::xxhash_64" (
        const table_view& input,
        const uint64_t seed
    ) except +
