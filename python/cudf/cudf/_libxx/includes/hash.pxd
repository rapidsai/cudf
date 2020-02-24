# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *

cdef extern from "cudf/hashing.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[table], vector[size_type]] \
        hash_partition "cudf::hash_partition" (
        const table_view& input,
        const vector[size_type]& columns_to_hash,
        int num_partitions
    ) except +

    cdef unique_ptr[column] hash "cudf::hash" (
        const table_view& input,
        const vector[uint32_t]& initial_hash
    ) except +
