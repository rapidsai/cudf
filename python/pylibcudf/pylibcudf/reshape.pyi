# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table

def interleave_columns(
    source_table: Table, stream: Stream | None = None
) -> Column: ...
def tile(
    source_table: Table, count: int, stream: Stream | None = None
) -> Table: ...
def table_to_array(
    input_table: Table,
    ptr: int,
    size: int,
    stream: Stream,
) -> None: ...
