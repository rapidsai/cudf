# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

cimport cudf._libxx.includes.types as cudf_types
from cudf._libxx.includes.column.column cimport column
from cudf._libxx.includes.table.table cimport table
from cudf._libxx.includes.table.table_view cimport table_view


cdef extern from "cudf/hashing.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[table], vector[cudf_types.size_type]] \
        hash_partition "cudf::hash_partition" (
        const table_view& input,
        const vector[cudf_types.size_type]& columns_to_hash,
        int num_partitions
    ) except +

    cdef unique_ptr[column] hash "cudf::hash" (
        const table_view& input,
        const vector[uint32_t]& initial_hash
    ) except +
