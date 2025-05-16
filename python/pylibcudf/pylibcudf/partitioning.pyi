# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.table import Table

def hash_partition(
    input: Table, columns_to_hash: list[int], num_partitions: int
) -> tuple[Table, list[int]]: ...
def partition(
    t: Table, partition_map: Column, num_partitions: int
) -> tuple[Table, list[int]]: ...
def round_robin_partition(
    input: Table, num_partitions: int, start_partition: int = 0
) -> tuple[Table, list[int]]: ...
