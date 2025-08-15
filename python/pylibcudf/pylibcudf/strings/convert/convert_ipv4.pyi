# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def ipv4_to_integers(
    input: Column, stream: Stream | None = None
) -> Column: ...
def integers_to_ipv4(
    integers: Column, stream: Stream | None = None
) -> Column: ...
def is_ipv4(input: Column, stream: Stream | None = None) -> Column: ...
