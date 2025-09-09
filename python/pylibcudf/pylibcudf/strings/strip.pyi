# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.side_type import SideType

def strip(
    input: Column,
    side: SideType = SideType.BOTH,
    to_strip: Scalar | None = None,
    stream: Stream | None = None,
) -> Column: ...
