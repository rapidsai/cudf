# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.partitioning cimport (
    partition as cpp_partition,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.utils cimport columns_from_unique_ptr, table_view_from_columns

from cudf._lib.reduce import minmax
from cudf._lib.stream_compaction import distinct_count as cpp_distinct_count

cimport cudf._lib.pylibcudf.libcudf.types as libcudf_types


@acquire_spill_lock()
def partition(list source_columns, Column partition_map,
              object num_partitions):
    """Partition source columns given a partitioning map

    Parameters
    ----------
    source_columns: list[Column]
        Columns to partition
    partition_map: Column
        Column of integer values that map each row in the input to a
        partition
    num_partitions: Optional[int]
        Number of output partitions (deduced from unique values in
        partition_map if None)

    Returns
    -------
    Pair of reordered columns and partition offsets

    Raises
    ------
    ValueError
        If the partition map has invalid entries (not all in [0,
        num_partitions)).
    """

    if num_partitions is None:
        num_partitions = cpp_distinct_count(partition_map, ignore_nulls=True)
    cdef int c_num_partitions = num_partitions
    cdef table_view c_source_view = table_view_from_columns(source_columns)

    cdef column_view c_partition_map_view = partition_map.view()

    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result
    if partition_map.size > 0:
        lo, hi = minmax(partition_map)
        if lo < 0 or hi >= num_partitions:
            raise ValueError("Partition map has invalid values")
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
