# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.types import DataType, MaskState, TypeId

def make_empty_column(
    type_or_id: DataType | TypeId,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_numeric_column(
    type_: DataType,
    size: int,
    mstate: MaskState,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_fixed_point_column(
    type_: DataType,
    size: int,
    mstate: MaskState,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_timestamp_column(
    type_: DataType,
    size: int,
    mstate: MaskState,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_duration_column(
    type_: DataType,
    size: int,
    mstate: MaskState,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def make_fixed_width_column(
    type_: DataType,
    size: int,
    mstate: MaskState,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
