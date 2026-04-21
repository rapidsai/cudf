# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from libc.stdint cimport uint32_t

from pylibcudf.libcudf.partitioning cimport hash_id, DEFAULT_HASH_SEED
from .column cimport Column
from .table cimport Table

ctypedef fused TableOrList:
    Table
    list

cpdef tuple[Table, list] hash_partition(
    Table input,
    TableOrList keys,
    int num_partitions,
    hash_id hash_function = *,
    uint32_t seed = *,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef tuple[Table, list] partition(
    Table t,
    Column partition_map,
    int num_partitions,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef tuple[Table, list] round_robin_partition(
    Table input,
    int num_partitions,
    int start_partition=*,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)
