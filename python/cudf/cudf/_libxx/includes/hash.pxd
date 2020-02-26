# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *

from cudf._libxx.includes.column.column cimport column
from cudf._libxx.includes.table.table cimport table
from cudf._libxx.includes.table.table_view cimport table_view

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
