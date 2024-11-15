# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.types import DataType

def to_fixed_point(input: Column, output_type: DataType) -> Column: ...
def from_fixed_point(input: Column) -> Column: ...
def is_fixed_point(
    input: Column, decimal_type: DataType | None = None
) -> Column: ...
