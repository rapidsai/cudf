# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.strings.side_type import SideType

def pad(
    input: Column,
    width: int,
    side: SideType,
    fill_char: str,
    stream: Stream | None = None,
) -> Column: ...
def zfill(
    input: Column, width: int, stream: Stream | None = None
) -> Column: ...
def zfill_by_widths(
    input: Column, widths: Column, stream: Stream | None = None
) -> Column: ...
