# Copyright (c) 2024, NVIDIA CORPORATION.
from pylibcudf.column import Column

def label_bins(
    input: Column,
    left_edges: Column,
    left_inclusive: bool,
    right_edges: Column,
    right_inclusive: bool,
) -> Column: ...
