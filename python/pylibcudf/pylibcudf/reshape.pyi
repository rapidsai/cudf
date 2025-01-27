# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.table import Table

def interleave_columns(source_table: Table) -> Column: ...
def tile(source_table: Table, count: int) -> Table: ...
