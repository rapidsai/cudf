# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
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
    mr: DeviceMemoryResource | None = None,
) -> Scalar: ...
def scan(
    col: Column,
    agg: Aggregation,
    inclusive: ScanType,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def minmax(
    col: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[Scalar, Scalar]: ...
def is_valid_reduce_aggregation(
    source: DataType, agg: Aggregation
) -> bool: ...
