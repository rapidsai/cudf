# Copyright (c) 2024, NVIDIA CORPORATION.

from collections.abc import Sequence

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.types import Interpolation, NullOrder, Order, Sorted

def quantile(
    input: Column,
    q: Sequence[float],
    interp: Interpolation = Interpolation.LINEAR,
    ordered_indices: Column | None = None,
    exact: bool = True,
    stream: Stream | None = None,
) -> Column: ...
def quantiles(
    input: Table,
    q: Sequence[float],
    interp: Interpolation = Interpolation.NEAREST,
    is_input_sorted: Sorted = Sorted.NO,
    column_order: list[Order] | None = None,
    null_precedence: list[NullOrder] | None = None,
    stream: Stream | None = None,
) -> Table: ...
