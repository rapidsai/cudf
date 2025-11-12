# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.gpumemoryview import gpumemoryview
from pylibcudf.types import MaskState

def copy_bitmask(
    col: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> DeviceBuffer: ...
def bitmask_allocation_size_bytes(number_of_bits: int) -> int: ...
def create_null_mask(
    size: int,
    state: MaskState = MaskState.UNINITIALIZED,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> DeviceBuffer: ...
def bitmask_and(
    columns: list[Column],
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[DeviceBuffer, int]: ...
def bitmask_or(
    columns: list[Column],
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> tuple[DeviceBuffer, int]: ...
def null_count(
    bitmask: gpumemoryview, start: int, stop: int, stream: Stream | None = None
) -> int: ...
