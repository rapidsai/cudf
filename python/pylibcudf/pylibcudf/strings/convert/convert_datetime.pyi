# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.types import DataType

def to_timestamps(
    input: Column,
    timestamp_type: DataType,
    format: str,
    stream: Stream | None = None,
) -> Column: ...
def from_timestamps(
    timestamps: Column,
    format: str,
    input_strings_names: Column,
    stream: Stream | None = None,
) -> Column: ...
def is_timestamp(
    input: Column, format: str, stream: Stream | None = None
) -> Column: ...
