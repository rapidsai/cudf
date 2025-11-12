# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

cimport pylibcudf.libcudf.types as libcudf_types
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf cimport partitioning as cpp_partitioning
from pylibcudf.libcudf.table.table cimport table
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource

__all__ = [
    "hash_partition",
    "partition",
    "round_robin_partition",
]

cpdef tuple[Table, list] hash_partition(
    Table input,
    list columns_to_hash,
    int num_partitions,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Partitions rows from the input table into multiple output tables.

    For details, see :cpp:func:`hash_partition`.

    Parameters
    ----------
    input : Table
        The table to partition
    columns_to_hash : list[int]
        Indices of input columns to hash
    num_partitions : int
        The number of partitions to use
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    tuple[Table, list[int]]
        An output table and a vector of row offsets to each partition
    """
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result
    cdef vector[libcudf_types.size_type] c_columns_to_hash = columns_to_hash
    cdef int c_num_partitions = num_partitions

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_partitioning.hash_partition(
            input.view(),
            c_columns_to_hash,
            c_num_partitions,
            cpp_partitioning.hash_id.HASH_MURMUR3,
            cpp_partitioning.DEFAULT_HASH_SEED,
            stream.view(),
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result.first), stream, mr), list(c_result.second)

cpdef tuple[Table, list] partition(
    Table t,
    Column partition_map,
    int num_partitions,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Partitions rows of `t` according to the mapping specified by `partition_map`.

    For details, see :cpp:func:`partition`.

    Parameters
    ----------
    t : Table
        The table to partition
    partition_map : Column
        Non-nullable column of integer values that map each row
        in `t` to it's partition.
    num_partitions : int
        The total number of partitions
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    tuple[Table, list[int]]
        An output table and a list of row offsets to each partition
    """
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result
    cdef int c_num_partitions = num_partitions

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_partitioning.partition(
            t.view(),
            partition_map.view(),
            c_num_partitions,
            stream.view(),
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result.first), stream, mr), list(c_result.second)


cpdef tuple[Table, list] round_robin_partition(
    Table input,
    int num_partitions,
    int start_partition=0,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Round-robin partition.

    For details, see :cpp:func:`round_robin_partition`.

    Parameters
    ----------
    input : Table
        The input table to be round-robin partitioned
    num_partitions : int
        Number of partitions for the table
    start_partition : int, default 0
        Index of the 1st partition
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    tuple[Table, list[int]]
        The partitioned table and the partition offsets
        for each partition within the table.
    """
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result
    cdef int c_num_partitions = num_partitions
    cdef int c_start_partition = start_partition

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_partitioning.round_robin_partition(
            input.view(),
            c_num_partitions,
            c_start_partition,
            stream.view(),
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result.first), stream, mr), list(c_result.second)
