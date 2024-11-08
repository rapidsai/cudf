# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum

from pylibcudf.column import Column

class Inclusive(IntEnum):
    YES = ...
    NO = ...

def label_bins(
    input: Column,
    left_edges: Column,
    left_inclusive: Inclusive,
    right_edges: Column,
    right_inclusive: Inclusive,
) -> Column: ...
