# Copyright (c) 2024, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table

def concatenate[ColumnOrTable: (Column, Table)](
    objects: list[ColumnOrTable],
    stream: Stream | None = None,
) -> ColumnOrTable: ...
