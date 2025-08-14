# Copyright (c) 2024, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.aggregation import RankMethod
from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.types import NullOrder, NullPolicy, Order

def sorted_order(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
) -> Column: ...
def stable_sorted_order(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
) -> Column: ...
def rank(
    input_view: Column,
    method: RankMethod,
    column_order: Order,
    null_handling: NullPolicy,
    null_precedence: NullOrder,
    percentage: bool,
    stream: Stream | None = None,
) -> Column: ...
def is_sorted(
    tbl: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
) -> bool: ...
def segmented_sort_by_key(
    values: Table,
    keys: Table,
    segment_offsets: Column,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
) -> Table: ...
def stable_segmented_sort_by_key(
    values: Table,
    keys: Table,
    segment_offsets: Column,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
) -> Table: ...
def sort_by_key(
    values: Table,
    keys: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
) -> Table: ...
def stable_sort_by_key(
    values: Table,
    keys: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
) -> Table: ...
def sort(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
) -> Table: ...
def stable_sort(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
) -> Table: ...
def top_k(
    col: Column,
    k: int,
    sort_order: Order = Order.DESCENDING,
    stream: Stream | None = None,
) -> Column: ...
def top_k_order(
    col: Column,
    k: int,
    sort_order: Order = Order.DESCENDING,
    stream: Stream | None = None,
) -> Column: ...
