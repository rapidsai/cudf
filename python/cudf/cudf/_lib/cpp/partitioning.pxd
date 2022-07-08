# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view


cdef extern from "cudf/partitioning.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] \
        hash_partition "cudf::hash_partition" (
        const table_view& input,
        const vector[libcudf_types.size_type]& columns_to_hash,
        int num_partitions
    ) except +

    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] \
        partition "cudf::partition" (
        const table_view& t,
        const column_view& partition_map,
        int num_partitions
    ) except +
