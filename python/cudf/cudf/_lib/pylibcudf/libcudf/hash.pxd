# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view


cdef extern from "cudf/hashing.hpp" namespace "cudf::hashing" nogil:

    cdef unique_ptr[column] murmurhash3_x86_32 "cudf::hashing::murmurhash3_x86_32" (
        const table_view& input,
        const uint32_t seed
    ) except +

    cdef unique_ptr[column] md5 "cudf::hashing::md5" (
        const table_view& input
    ) except +

    cdef unique_ptr[column] sha1 "cudf::hashing::sha1" (
        const table_view& input
    ) except +

    cdef unique_ptr[column] sha224 "cudf::hashing::sha224" (
        const table_view& input
    ) except +

    cdef unique_ptr[column] sha256 "cudf::hashing::sha256" (
        const table_view& input
    ) except +

    cdef unique_ptr[column] sha384 "cudf::hashing::sha384" (
        const table_view& input
    ) except +

    cdef unique_ptr[column] sha512 "cudf::hashing::sha512" (
        const table_view& input
    ) except +

    cdef unique_ptr[column] xxhash_64 "cudf::hashing::xxhash_64" (
        const table_view& input,
        const uint64_t seed
    ) except +
