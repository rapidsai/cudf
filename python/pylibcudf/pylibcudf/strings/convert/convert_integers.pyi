# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.types import DataType

def to_integers(
    input: Column, output_type: DataType, stream: Stream | None = None
) -> Column: ...
def from_integers(
    integers: Column, stream: Stream | None = None
) -> Column: ...
def is_integer(
    input: Column,
    int_type: DataType | None = None,
    stream: Stream | None = None,
) -> Column: ...
def hex_to_integers(
    input: Column, output_type: DataType, stream: Stream | None = None
) -> Column: ...
def is_hex(input: Column, stream: Stream | None = None) -> Column: ...
def integers_to_hex(input: Column, stream: Stream | None = None) -> Column: ...
