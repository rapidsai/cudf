# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.types import NullOrder, Order

def lower_bound(
    haystack: Table,
    needles: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def upper_bound(
    haystack: Table,
    needles: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def contains(
    haystack: Column,
    needles: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
