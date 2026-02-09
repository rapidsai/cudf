# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table

def hash_partition(
    input: Table,
    columns_to_hash: list[int],
    num_partitions: int,
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
