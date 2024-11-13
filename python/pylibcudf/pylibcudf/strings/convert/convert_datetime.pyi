# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.types import DataType

def to_timestamps(
    input: Column, timestamp_type: DataType, format: str
) -> Column: ...
def from_timestamps(
    timestamps: Column, format: str, input_strings_names: Column
) -> Column: ...
def is_timestamp(input: Column, format: str) -> Column: ...
