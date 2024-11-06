# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum, auto

from pylibcudf.column import Column

class Inclusive(IntEnum):
    YES = auto()
    NO = auto()

def label_bins(
    input: Column,
    left_edges: Column,
    left_inclusive: Inclusive,
    right_edges: Column,
    right_inclusive: Inclusive,
) -> Column: ...
