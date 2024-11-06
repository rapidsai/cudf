# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum, auto

from pylibcudf.column import Column

class RoundingMethod(IntEnum):
    HALF_UP = auto()
    HALF_EVEN = auto()

def round(
    source: Column,
    decimal_places: int = 0,
    round_method: RoundingMethod = RoundingMethod.HALF_UP,
) -> Column: ...
