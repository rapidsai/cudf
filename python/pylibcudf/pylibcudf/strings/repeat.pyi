# Copyright (c) 2024, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def repeat_strings(
    input: Column,
    repeat_times: Column | int,
    stream: Stream | None = None,
) -> Column: ...
