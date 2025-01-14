# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def to_booleans(input: Column, true_string: Scalar) -> Column: ...
def from_booleans(
    booleans: Column, true_string: Scalar, false_string: Scalar
) -> Column: ...
