# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf as plc

from cudf._lib.reduce import minmax
from cudf._lib.stream_compaction import distinct_count as cpp_distinct_count


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

    if partition_map.size > 0:
        lo, hi = minmax(partition_map)
        if lo < 0 or hi >= num_partitions:
            raise ValueError("Partition map has invalid values")

    plc_table, offsets = plc.partitioning.partition(
        plc.Table([col.to_pylibcudf(mode="read") for col in source_columns]),
        partition_map.to_pylibcudf(mode="read"),
        num_partitions
    )
    return [Column.from_pylibcudf(col) for col in plc_table.columns()], offsets
