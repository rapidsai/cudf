# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum

from pylibcudf.column import Column

class RoundingMethod(IntEnum):
    HALF_UP = ...
    HALF_EVEN = ...

def round(
    source: Column,
    decimal_places: int = 0,
    round_method: RoundingMethod = RoundingMethod.HALF_UP,
) -> Column: ...
