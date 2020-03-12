# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
cimport cudf._libxx.cpp.types as libcudf_types


cdef extern from "cudf/partitioning.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] \
        hash_partition "cudf::hash_partition" (
        const table_view& input,
        const vector[libcudf_types.size_type]& columns_to_hash,
        int num_partitions
    ) except +
