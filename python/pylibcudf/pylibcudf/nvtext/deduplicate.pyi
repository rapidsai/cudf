# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.column import Column

def build_suffix_array(input: Column, min_width: int) -> Column: ...
def resolve_duplicates(
    input: Column, indices: Column, min_width: int
) -> Column: ...
def resolve_duplicates_pair(
    input1: Column,
    indices1: Column,
    input2: Column,
    indices2: Column,
    min_width: int,
) -> Column: ...
