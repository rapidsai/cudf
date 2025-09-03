# Copyright (c) 2024, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.table import Table

def transpose(input_table: Table, stream: Stream | None = None) -> Table: ...
