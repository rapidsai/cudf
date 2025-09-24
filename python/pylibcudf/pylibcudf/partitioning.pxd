# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .table cimport Table


cpdef tuple[Table, list] hash_partition(
    Table input,
    list columns_to_hash,
    int num_partitions,
    Stream stream = *
)

cpdef tuple[Table, list] partition(
    Table t,
    Column partition_map,
    int num_partitions,
    Stream stream = *,
)

cpdef tuple[Table, list] round_robin_partition(
    Table input,
    int num_partitions,
    int start_partition=*,
    Stream stream = *
)
