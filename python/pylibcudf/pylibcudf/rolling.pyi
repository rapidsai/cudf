# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.aggregation import Aggregation
from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table
from pylibcudf.types import DataType, NullOrder, Order

class Unbounded: ...
class CurrentRow: ...

class BoundedClosed:
    def __init__(self, delta: Scalar) -> None: ...
    delta: Scalar

class BoundedOpen:
    def __init__(self, delta: Scalar) -> None: ...
    delta: Scalar

class RollingRequest:
    def __init__(
        self, values: Column, min_periods: int, aggregation: Aggregation
    ) -> None: ...

RangeWindowType = BoundedClosed | BoundedOpen | CurrentRow | Unbounded

def grouped_range_rolling_window(
    group_keys: Table,
    orderby: Column,
    order: Order,
    null_order: NullOrder,
    preceding: RangeWindowType,
    following: RangeWindowType,
    requests: list[RollingRequest],
) -> Table: ...
def rolling_window[WindowType: (Column, int)](
    source: Column,
    preceding_window: WindowType,
    following_window: WindowType,
    min_periods: int,
    agg: Aggregation,
) -> Column: ...
def is_valid_rolling_aggregation(
    source: DataType, agg: Aggregation
) -> bool: ...
