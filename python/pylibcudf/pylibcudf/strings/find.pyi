# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def find(
    input: Column,
    target: Column | Scalar,
    start: int = 0,
    stop: int = -1,
    stream: Stream | None = None,
) -> Column: ...
def rfind(
    input: Column,
    target: Scalar,
    start: int = 0,
    stop: int = -1,
    stream: Stream | None = None,
) -> Column: ...
def contains(
    input: Column, target: Column | Scalar, stream: Stream | None = None
) -> Column: ...
def starts_with(
    input: Column, target: Column | Scalar, stream: Stream | None = None
) -> Column: ...
def ends_with(
    input: Column, target: Column | Scalar, stream: Stream | None = None
) -> Column: ...
