# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.types import DataType

def to_floats(
    strings: Column, output_type: DataType, stream: Stream | None = None
) -> Column: ...
def from_floats(floats: Column, stream: Stream | None = None) -> Column: ...
def is_float(input: Column, stream: Stream | None = None) -> Column: ...
