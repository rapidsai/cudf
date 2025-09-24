# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table

def find_multiple(
    input: Column, targets: Column, stream: Stream | None = None
) -> Column: ...
def contains_multiple(
    input: Column, targets: Column, stream: Stream | None = None
) -> Table: ...
