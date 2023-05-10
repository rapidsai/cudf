# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view


cdef extern from "cudf/hashing.hpp" namespace "cudf" nogil:

    ctypedef enum hash_id "cudf::hash_id":
        HASH_IDENTITY "cudf::hash_id::HASH_IDENTITY"
        HASH_MURMUR3 "cudf::hash_id::HASH_MURMUR3"
        HASH_SPARK_MURMUR3 "cudf::hash_id::HASH_SPARK_MURMUR3"
        HASH_MD5 "cudf::hash_id::HASH_MD5"

    cdef unique_ptr[column] hash "cudf::hash" (
        const table_view& input,
        const hash_id hash_function,
        const uint32_t seed
    ) except +
