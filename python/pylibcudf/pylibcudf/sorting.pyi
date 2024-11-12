# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.aggregation import RankMethod
from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.types import NullOrder, NullPolicy, Order

def sorted_order(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
) -> Column: ...
def stable_sorted_order(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
) -> Column: ...
def rank(
    input_view: Column,
    method: RankMethod,
    column_order: Order,
    null_handling: NullPolicy,
    null_precedence: NullOrder,
    percentage: bool,
) -> Column: ...
def is_sorted(
    tbl: Table, column_order: list[Order], null_precedence: list[NullOrder]
) -> bool: ...
def segmented_sort_by_key(
    values: Table,
    keys: Table,
    segment_offsets: Column,
    column_order: list[Order],
    null_precedence: list[NullOrder],
) -> Table: ...
def stable_segmented_sort_by_key(
    values: Table,
    keys: Table,
    segment_offsets: Column,
    column_order: list[Order],
    null_precedence: list[NullOrder],
) -> Table: ...
def sort_by_key(
    values: Table,
    keys: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
) -> Table: ...
def stable_sort_by_key(
    values: Table,
    keys: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
) -> Table: ...
def sort(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
) -> Table: ...
def stable_sort(
    source_table: Table,
    column_order: list[Order],
    null_precedence: list[NullOrder],
) -> Table: ...
