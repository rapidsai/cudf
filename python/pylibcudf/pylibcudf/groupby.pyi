# Copyright (c) 2024, NVIDIA CORPORATION.

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
        self, requests: list[GroupByRequest]
    ) -> tuple[Table, list[Table]]: ...
    def scan(
        self, requests: list[GroupByRequest]
    ) -> tuple[Table, list[Table]]: ...
    def shift(
        self, values: Table, offset: list[int], fill_values: list[Scalar]
    ) -> tuple[Table, Table]: ...
    def replace_nulls(
        self, value: Table, replace_policies: list[ReplacePolicy]
    ) -> tuple[Table, Table]: ...
    def get_groups(
        self, values: Table | None = None
    ) -> tuple[list[int], Table, Table]: ...
