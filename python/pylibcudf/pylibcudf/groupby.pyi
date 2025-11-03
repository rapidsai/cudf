# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.aggregation import Aggregation
from pylibcudf.column import Column
from pylibcudf.replace import ReplacePolicy
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table
from pylibcudf.types import NullOrder, NullPolicy, Order, Sorted

class GroupByRequest:
    def __init__(
        self, values: Column, aggregations: list[Aggregation]
    ) -> None: ...

class GroupBy:
    def __init__(
        self,
        keys: Table,
        null_handling: NullPolicy = NullPolicy.EXCLUDE,
        keys_are_sorted: Sorted = Sorted.NO,
        column_order: list[Order] | None = None,
        null_precedence: list[NullOrder] | None = None,
    ) -> None: ...
    def aggregate(
        self,
        requests: list[GroupByRequest],
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> tuple[Table, list[Table]]: ...
    def scan(
        self,
        requests: list[GroupByRequest],
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> tuple[Table, list[Table]]: ...
    def shift(
        self,
        values: Table,
        offset: list[int],
        fill_values: list[Scalar],
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> tuple[Table, Table]: ...
    def replace_nulls(
        self,
        value: Table,
        replace_policies: list[ReplacePolicy],
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> tuple[Table, Table]: ...
    def get_groups(
        self,
        values: Table | None = None,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> tuple[list[int], Table, Table]: ...
