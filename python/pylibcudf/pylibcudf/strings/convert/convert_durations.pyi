# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.types import DataType

def to_durations(
    input: Column, duration_type: DataType, format: str
) -> Column: ...
def from_durations(durations: Column, format: str | None = None) -> Column: ...
