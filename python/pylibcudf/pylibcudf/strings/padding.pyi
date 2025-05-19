# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.strings.side_type import SideType

def pad(
    input: Column, width: int, side: SideType, fill_char: str
) -> Column: ...
def zfill(input: Column, width: int) -> Column: ...
