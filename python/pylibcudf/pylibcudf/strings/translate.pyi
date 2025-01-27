# Copyright (c) 2024, NVIDIA CORPORATION.
from collections.abc import Mapping
from enum import IntEnum

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

class FilterType(IntEnum):
    KEEP = ...
    REMOVE = ...

def translate(
    input: Column, chars_table: Mapping[int | str, int | str]
) -> Column: ...
def filter_characters(
    input: Column,
    characters_to_filter: Mapping[int | str, int | str],
    keep_characters: FilterType,
    replacement: Scalar,
) -> Column: ...
