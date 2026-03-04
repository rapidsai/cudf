# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def build_suffix_array(
    input: Column,
    min_width: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def resolve_duplicates(
    input: Column,
    indices: Column,
    min_width: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def resolve_duplicates_pair(
    input1: Column,
    indices1: Column,
    input2: Column,
    indices2: Column,
    min_width: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
