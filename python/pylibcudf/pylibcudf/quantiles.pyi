# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.types import Interpolation, NullOrder, Order, Sorted

def quantile(
    input: Column,
    q: Iterable[float],
    interp: Interpolation = Interpolation.LINEAR,
    ordered_indices: Column | None = None,
    exact: bool = True,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def quantiles(
    input: Table,
    q: Iterable[float],
    interp: Interpolation = Interpolation.NEAREST,
    is_input_sorted: Sorted = Sorted.NO,
    column_order: list[Order] | None = None,
    null_precedence: list[NullOrder] | None = None,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
