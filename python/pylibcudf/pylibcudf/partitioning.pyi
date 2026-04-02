# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table

class HashId(IntEnum):
    HASH_IDENTITY = ...
    HASH_MURMUR3 = ...

def hash_partition(
    input: Table,
    keys: Table | list[int],
    num_partitions: int,
    hash_function: HashId = ...,
    seed: int = ...,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Table, list[int]]: ...
def partition(
    t: Table,
    partition_map: Column,
    num_partitions: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Table, list[int]]: ...
def round_robin_partition(
    input: Table,
    num_partitions: int,
    start_partition: int = 0,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Table, list[int]]: ...
