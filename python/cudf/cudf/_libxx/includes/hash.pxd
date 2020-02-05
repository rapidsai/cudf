# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.includes.lib cimport *

cdef extern from "cudf/hashing.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[table], vector[size_type]] hash_partition "cudf::hash_partition" (
        table_view input,
        vector[size_type] columns_to_hash,
        int num_partitions
    ) except +

    cdef unique_ptr[column] hash "cudf::hash" (
        table_view input,
        vector[uint32_t] initial_hash
    ) except +
