# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.partitioning cimport partition as cpp_partition
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.utils cimport columns_from_unique_ptr, table_view_from_columns
from cudf._lib.stream_compaction import distinct_count as cpp_distinct_count

cimport cudf._lib.cpp.types as libcudf_types


def partition(list source_columns, Column partition_map,
              object num_partitions):

    if num_partitions is None:
        num_partitions = cpp_distinct_count(partition_map, ignore_nulls=True)
    cdef int c_num_partitions = num_partitions
    cdef table_view c_source_view = table_view_from_columns(source_columns)

    cdef column_view c_partition_map_view = partition_map.view()

    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result
    with nogil:
        c_result = move(
            cpp_partition(
                c_source_view,
                c_partition_map_view,
                c_num_partitions
            )
        )

    return (
        columns_from_unique_ptr(move(c_result.first)), list(c_result.second)
    )
