# Copyright (c) 2020-2024, NVIDIA CORPORATION.
cimport pylibcudf.libcudf.types as libcudf_types
from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view


cdef extern from "cudf/partitioning.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] \
        hash_partition "cudf::hash_partition" (
        const table_view& input,
        const vector[libcudf_types.size_type]& columns_to_hash,
        int num_partitions
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] \
        partition "cudf::partition" (
        const table_view& t,
        const column_view& partition_map,
        int num_partitions
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] \
        round_robin_partition "cudf::round_robin_partition" (
        const table_view& input,
        int num_partitions,
        int start_partition
    ) except +libcudf_exception_handler
