# Copyright (c) 2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def reverse(input: Column, stream: Stream | None = None) -> Column: ...
