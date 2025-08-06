# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum

from rmm.pylibrmm.stream import Stream

from pylibcudf.aggregation import Aggregation
from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.types import DataType

class ScanType(IntEnum):
    INCLUSIVE = ...
    EXCLUSIVE = ...

def reduce(
    col: Column,
    agg: Aggregation,
    data_type: DataType,
    stream: Stream | None = None,
) -> Scalar: ...
def scan(
    col: Column,
    agg: Aggregation,
    inclusive: ScanType,
    stream: Stream | None = None,
) -> Column: ...
def minmax(
    col: Column, stream: Stream | None = None
) -> tuple[Scalar, Scalar]: ...
