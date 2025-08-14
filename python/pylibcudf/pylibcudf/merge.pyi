# Copyright (c) 2024, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.table import Table
from pylibcudf.types import NullOrder, Order

def merge(
    tables_to_merge: list[Table],
    key_cols: list[int],
    column_order: list[Order],
    null_precedence: list[NullOrder],
    stream: Stream | None = None,
) -> Table: ...
