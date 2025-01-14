# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.side_type import SideType

def strip(
    input: Column,
    side: SideType = SideType.BOTH,
    to_strip: Scalar | None = None,
) -> Column: ...
