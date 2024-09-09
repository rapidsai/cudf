# Copyright (c) 2024, NVIDIA CORPORATION.

cimport pylibcudf.libcudf.types as libcudf_types
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf cimport partitioning as cpp_partitioning
from pylibcudf.libcudf.table.table cimport table

from .column cimport Column
from .table cimport Table


cpdef tuple[Table, list] hash_partition(
    Table input,
    list columns_to_hash,
    int num_partitions
):
    """
    Partitions rows from the input table into multiple output tables.

    Parameters
    ----------
    input : Table
        The table to partition
    columns_to_hash : list[int]
        Indices of input columns to hash
    num_partitions : int
        The number of partitions to use

    Returns
    -------
    tuple[Table, list[int]]
        An output table and a vector of row offsets to each partition
    """
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result

    with nogil:
        c_result = move(
            cpp_partitioning.hash_partition(
                table.view(), columns_to_hash, num_partitions
            )
        )

    return Table.from_libcudf(move(c_result.first)), c_result.second

cpdef tuple[Table, list] partition(Table t, Column partition_map, int num_partitions):
    """
    Partitions rows of `t` according to the mapping specified by `partition_map`.

    Parameters
    ----------
    t : Table
        The table to partition
    partition_map : list[int]
        Non-nullable column of integer values that map each row
        in `t` to it's partition.
    num_partitions : int
        The total number of partitions

    Returns
    -------
    tuple[Table, list[int]]
        An output table and a list of row offsets to each partition
    """
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result

    with nogil:
        c_result = move(
            cpp_partitioning.partition(t.view(), partition_map.view(), num_partitions)
        )

    return Table.from_libcudf(move(c_result.first)), c_result.second


cpdef tuple[Table, list] round_robin_partition(
    Table input,
    int num_partitions,
    int start_partition=0
):
    """
    Round-robin partition.

    Parameters
    ----------
    input : Table
        The input table to be round-robin partitioned
    num_partitions : int
        Number of partitions for the table
    start_partition : int, default 0
        Index of the 1st partition

    Returns
    -------
    tuple[Table, list[int]]
        The partitioned table and the partition offsets
        for each partition within the table.
    """
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result

    with nogil:
        c_result = move(
            cpp_partitioning.round_robin_partition(
                table.view(), num_partitions, start_partition
            )
        )

    return Table.from_libcudf(move(c_result.first)), c_result.second
