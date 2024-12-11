# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.types import NullOrder, Order

def lower_bound(
    haystack: Table,
    needles: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
) -> Column: ...
def upper_bound(
    haystack: Table,
    needles: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
) -> Column: ...
def contains(haystack: Column, needles: Column) -> Column: ...
