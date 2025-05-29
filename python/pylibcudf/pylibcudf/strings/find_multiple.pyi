# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.table import Table

def find_multiple(input: Column, targets: Column) -> Column: ...
def contains_multiple(input: Column, targets: Column) -> Table: ...
