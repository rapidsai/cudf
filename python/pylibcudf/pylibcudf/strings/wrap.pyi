# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def wrap(
    input: Column, width: int, stream: Stream | None = None
) -> Column: ...
