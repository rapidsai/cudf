# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.table import Table

def concatenate[ColumnOrTable: (Column, Table)](
    objects: list[ColumnOrTable],
) -> ColumnOrTable: ...
