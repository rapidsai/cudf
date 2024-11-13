# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

class ReplacePolicy(IntEnum):
    PRECEDING = ...
    FOLLOWING = ...

def replace_nulls(
    source_column: Column, replacement: Column | Scalar | ReplacePolicy
) -> Column: ...
def find_and_replace_all(
    source_column: Column,
    values_to_replace: Column,
    replacement_values: Column,
) -> Column: ...
def clamp(
    source_column: Column,
    lo: Scalar,
    hi: Scalar,
    lo_replace: Scalar | None = None,
    hi_replace: Scalar | None = None,
) -> Column: ...
def normalize_nans_and_zeros(
    source_column: Column, inplace: bool = False
) -> Column: ...
