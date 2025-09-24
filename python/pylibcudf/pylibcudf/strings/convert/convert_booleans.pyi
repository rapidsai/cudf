# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def to_booleans(
    input: Column, true_string: Scalar, stream: Stream | None = None
) -> Column: ...
def from_booleans(
    booleans: Column,
    true_string: Scalar,
    false_string: Scalar,
    stream: Stream | None = None,
) -> Column: ...
